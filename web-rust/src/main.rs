//! Video Tagger Web Server - Rust implementation
//!
//! A fast web server for viewing and editing video tags.
//! Shares the same database and API as the Python version.

use axum::{
    Router,
    extract::{Path, Query, State},
    http::{StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, put},
    Json,
};
use clap::Parser;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashSet, HashMap},
    path::PathBuf,
    sync::Arc,
    time::Duration,
    process::Command,
};
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::RwLock;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
};
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Video Tagger Web UI - Rust Edition
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind to
    #[arg(long, default_value_t = 5001)]
    port: u16,

    /// Video directory (optional)
    #[arg(long)]
    dir: Option<PathBuf>,
}

/// Thumbnail generation status for a video
#[derive(Clone, Debug)]
enum ThumbnailStatus {
    Pending,
    Generating,
    Ready(PathBuf),
    Failed,
}

/// Shared application state
struct AppState {
    db_path: PathBuf,
    cache_dir: PathBuf,
    video_extensions: HashSet<String>,
    /// Map from video path to thumbnail status
    thumbnails: RwLock<HashMap<String, ThumbnailStatus>>,
}

impl AppState {
    fn new() -> Self {
        let base_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();

        let db_path = base_dir.join("data").join("faces.db");
        let cache_dir = base_dir.join(".cache").join("thumbnails");

        // Create cache directory if it doesn't exist
        std::fs::create_dir_all(&cache_dir).ok();

        let video_extensions: HashSet<String> = [
            ".mp4", ".mov", ".avi", ".mkv", ".m4v"
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            db_path,
            cache_dir,
            video_extensions,
            thumbnails: RwLock::new(HashMap::new()),
        }
    }

    fn get_db(&self) -> Result<Connection, rusqlite::Error> {
        Connection::open(&self.db_path)
    }

    /// Get the thumbnail path for a video
    fn thumbnail_path(&self, video_path: &str) -> PathBuf {
        // Create a hash of the video path for the filename
        let hash = format!("{:x}", md5_hash(video_path));
        self.cache_dir.join(format!("{}.jpg", hash))
    }
}

/// Simple hash function for creating thumbnail filenames
fn md5_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

#[derive(Serialize)]
struct Video {
    path: String,
    filename: String,
    content_tags: Vec<String>,
    person_tags: Vec<String>,
    processed_at: String,
}

#[derive(Serialize)]
struct Person {
    id: i64,
    name: String,
}

#[derive(Deserialize)]
struct RenameRequest {
    name: String,
}

#[derive(Serialize)]
struct RenameResponse {
    success: bool,
    old_name: String,
    new_name: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Deserialize)]
struct VideoQuery {
    path: String,
}

#[tokio::main]
async fn main() {
    // Initialize tracing subscriber
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_level(true)
                .compact()
        )
        .with(
            tracing_subscriber::filter::Targets::new()
                .with_target("tower_http::trace", Level::DEBUG)
                .with_target("web_rust", Level::DEBUG)
                .with_default(Level::INFO)
        )
        .init();

    let args = Args::parse();

    let state = Arc::new(AppState::new());

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/videos", get(api_videos))
        .route("/api/persons", get(api_persons))
        .route("/api/persons/:person_id", put(api_rename_person))
        .route("/api/video", get(api_serve_video))
        .route("/api/thumbnail", get(api_thumbnail))
        .route("/video/*video_path", get(video_page_handler))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &axum::http::Request<_>| {
                    tracing::info_span!(
                        "http",
                        method = %request.method(),
                        uri = %request.uri().path(),
                    )
                })
                .on_response(|response: &axum::http::Response<_>, latency: Duration, _span: &tracing::Span| {
                    tracing::info!(
                        status = %response.status().as_u16(),
                        latency_ms = %latency.as_millis(),
                        "response"
                    );
                })
        )
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    // Spawn background thumbnail generation task
    let thumbnail_state = state.clone();
    tokio::spawn(async move {
        generate_missing_thumbnails(thumbnail_state).await;
    });

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

    println!("\n🎬 Video Tagger Web UI (Rust)");
    println!("   Open http://{} in your browser\n", addr);

    axum::serve(listener, app).await.unwrap();
}

/// Serve the index page
async fn index_handler() -> Html<String> {
    eprintln!("GET /");
    Html(INDEX_HTML.to_string())
}

/// Serve a video detail page
async fn video_page_handler(Path(video_path): Path<String>) -> Html<String> {
    eprintln!("GET /video/{}", video_path);
    let video_path = urlencoding::decode(&video_path).unwrap_or_default().to_string();
    let html = VIDEO_HTML.replace("{{ video_path }}", &video_path);
    Html(html)
}

/// Get all processed videos with their tags
async fn api_videos(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Video>>, StatusCode> {
    eprintln!("GET /api/videos");
    let conn = state.get_db().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut stmt = conn
        .prepare(
            "SELECT video_path, content_tags, person_tags, processed_at
             FROM video_tags
             ORDER BY processed_at DESC"
        )
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let videos: Vec<Video> = stmt
        .query_map([], |row| {
            let path: String = row.get(0)?;
            let content_tags_str: Option<String> = row.get(1)?;
            let person_tags_str: Option<String> = row.get(2)?;
            let processed_at: String = row.get(3)?;

            let filename = std::path::Path::new(&path)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let content_tags: Vec<String> = content_tags_str
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            let person_tags: Vec<String> = person_tags_str
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();

            Ok(Video {
                path,
                filename,
                content_tags,
                person_tags,
                processed_at,
            })
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(|r| r.ok())
        .filter(|v| std::path::Path::new(&v.path).exists())
        .collect();

    Ok(Json(videos))
}

/// Get all known persons
async fn api_persons(State(state): State<Arc<AppState>>) -> Result<Json<Vec<Person>>, StatusCode> {
    eprintln!("GET /api/persons");
    let conn = state.get_db().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut stmt = conn
        .prepare("SELECT id, name FROM persons ORDER BY id")
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let persons: Vec<Person> = stmt
        .query_map([], |row| {
            Ok(Person {
                id: row.get(0)?,
                name: row.get(1)?,
            })
        })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(Json(persons))
}

/// Rename a person
async fn api_rename_person(
    State(state): State<Arc<AppState>>,
    Path(person_id): Path<i64>,
    Json(req): Json<RenameRequest>,
) -> Result<Json<RenameResponse>, (StatusCode, Json<ErrorResponse>)> {
    eprintln!("PUT /api/persons/{}", person_id);
    let new_name = req.name.trim().to_string();

    if new_name.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: "Name is required".to_string() }),
        ));
    }

    let conn = state.get_db().map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Database error".to_string() }),
        )
    })?;

    // Get old name
    let old_name: String = conn
        .query_row(
            "SELECT name FROM persons WHERE id = ?",
            [person_id],
            |row| row.get(0),
        )
        .map_err(|_| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse { error: "Person not found".to_string() }),
            )
        })?;

    // Update person name
    conn.execute(
        "UPDATE persons SET name = ? WHERE id = ?",
        rusqlite::params![&new_name, person_id],
    )
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Failed to update".to_string() }),
        )
    })?;

    // Update all video tags that reference this person
    let mut stmt = conn
        .prepare("SELECT video_path, person_tags FROM video_tags WHERE person_tags LIKE ?")
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: "Database error".to_string() }),
            )
        })?;

    let pattern = format!("%{}%", old_name);
    let updates: Vec<(String, String)> = stmt
        .query_map([&pattern], |row| {
            let path: String = row.get(0)?;
            let tags: String = row.get::<_, Option<String>>(1)?.unwrap_or_default();
            Ok((path, tags))
        })
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: "Database error".to_string() }),
            )
        })?
        .filter_map(|r| r.ok())
        .collect();

    for (path, tags) in updates {
        let updated_tags: Vec<&str> = tags
            .split(',')
            .map(|t| if t == old_name { new_name.as_str() } else { t })
            .collect();
        let updated_tags_str = updated_tags.join(",");

        conn.execute(
            "UPDATE video_tags SET person_tags = ? WHERE video_path = ?",
            rusqlite::params![&updated_tags_str, &path],
        )
        .ok();
    }

    Ok(Json(RenameResponse {
        success: true,
        old_name,
        new_name,
    }))
}

/// Serve a video file
async fn api_serve_video(
    State(state): State<Arc<AppState>>,
    Query(query): Query<VideoQuery>,
) -> Result<Response, StatusCode> {
    eprintln!("GET /api/video?path={}", query.path);
    let video_path = urlencoding::decode(&query.path)
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .to_string();

    let path = std::path::Path::new(&video_path);

    if !path.exists() {
        return Err(StatusCode::NOT_FOUND);
    }

    // Security: check extension
    let ext = path
        .extension()
        .map(|s| format!(".{}", s.to_string_lossy().to_lowercase()))
        .unwrap_or_default();

    if !state.video_extensions.contains(&ext) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Read the file
    let mut file = File::open(&video_path)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;

    let mut contents = Vec::new();
    file.read_to_end(&mut contents)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Determine MIME type
    let mime_type = match ext.as_str() {
        ".mp4" => "video/mp4",
        ".mov" => "video/quicktime",
        ".avi" => "video/x-msvideo",
        ".mkv" => "video/x-matroska",
        ".m4v" => "video/x-m4v",
        _ => "video/mp4",
    };

    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, mime_type)],
        contents,
    ).into_response())
}

/// Serve a video thumbnail
async fn api_thumbnail(
    State(state): State<Arc<AppState>>,
    Query(query): Query<VideoQuery>,
) -> Result<Response, StatusCode> {
    let video_path = urlencoding::decode(&query.path)
        .map_err(|_| StatusCode::BAD_REQUEST)?
        .to_string();

    let thumbnail_path = state.thumbnail_path(&video_path);

    // Check if thumbnail exists
    if thumbnail_path.exists() {
        let mut file = File::open(&thumbnail_path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;

        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        return Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg")],
            contents,
        ).into_response());
    }

    // Generate thumbnail on-the-fly if it doesn't exist
    let video_path_clone = video_path.clone();
    let thumbnail_path_clone = thumbnail_path.clone();

    let result = tokio::task::spawn_blocking(move || {
        generate_thumbnail(&video_path_clone, &thumbnail_path_clone)
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if result {
        let mut file = File::open(&thumbnail_path)
            .await
            .map_err(|_| StatusCode::NOT_FOUND)?;

        let mut contents = Vec::new();
        file.read_to_end(&mut contents)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        return Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "image/jpeg")],
            contents,
        ).into_response());
    }

    Err(StatusCode::NOT_FOUND)
}

/// Generate a thumbnail for a video using ffmpeg
fn generate_thumbnail(video_path: &str, thumbnail_path: &PathBuf) -> bool {
    // Use ffmpeg to extract a frame at 1 second (or 10% into video)
    let output = Command::new("ffmpeg")
        .args([
            "-y",                          // Overwrite output
            "-loglevel", "error",          // Suppress verbose output
            "-ss", "1",                    // Seek to 1 second
            "-i", video_path,              // Input file
            "-vframes", "1",               // Extract 1 frame
            "-vf", "scale=320:-1",         // Scale to 320px width
            "-q:v", "8",                   // JPEG quality (lower = better, 2-31)
            thumbnail_path.to_str().unwrap_or(""),
        ])
        .output();

    match output {
        Ok(result) => {
            if !result.status.success() {
                eprintln!("ffmpeg failed for {}: {}", video_path, String::from_utf8_lossy(&result.stderr));
            }
            result.status.success()
        },
        Err(e) => {
            eprintln!("ffmpeg error for {}: {}", video_path, e);
            false
        }
    }
}

/// Background task to generate thumbnails for all videos in the database
async fn generate_missing_thumbnails(state: Arc<AppState>) {
    // Small delay to let the server start
    tokio::time::sleep(Duration::from_millis(500)).await;

    let videos: Vec<String> = {
        let conn = match state.get_db() {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut stmt = match conn.prepare("SELECT video_path FROM video_tags") {
            Ok(s) => s,
            Err(_) => return,
        };

        stmt.query_map([], |row| row.get(0))
            .ok()
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    };

    let total = videos.len();
    let mut generated = 0;
    let mut skipped = 0;

    println!("🖼️  Checking thumbnails for {} videos...", total);

    for video_path in videos {
        let thumbnail_path = state.thumbnail_path(&video_path);

        // Skip if thumbnail already exists
        if thumbnail_path.exists() {
            skipped += 1;
            continue;
        }

        // Skip if video doesn't exist
        if !std::path::Path::new(&video_path).exists() {
            continue;
        }

        // Generate thumbnail in blocking task
        let video_path_clone = video_path.clone();
        let thumbnail_path_clone = thumbnail_path.clone();

        let result = tokio::task::spawn_blocking(move || {
            generate_thumbnail(&video_path_clone, &thumbnail_path_clone)
        })
        .await;

        if result.is_ok() && result.unwrap() {
            generated += 1;
            if generated % 10 == 0 {
                println!("🖼️  Generated {} thumbnails...", generated);
            }
        }
    }

    if generated > 0 {
        println!("🖼️  Generated {} new thumbnails ({} already existed)", generated, skipped);
    } else if skipped > 0 {
        println!("🖼️  All {} thumbnails up to date", skipped);
    }
}

// Embedded HTML templates (same as Python version)
const INDEX_HTML: &str = include_str!("../templates/index.html");
const VIDEO_HTML: &str = include_str!("../templates/video.html");
