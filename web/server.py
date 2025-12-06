#!/usr/bin/env python3
"""Simple web server for viewing and editing video tags."""

import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, jsonify, render_template, request, send_file, abort

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FACES_DB_PATH, VIDEO_EXTENSIONS
from storage.database import FaceDatabase

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Will be set when server starts
VIDEO_DIR: Path | None = None


def get_db():
    """Get a database connection."""
    return FaceDatabase(FACES_DB_PATH)


@app.route('/')
def index():
    """Main page showing all processed videos."""
    return render_template('index.html')


@app.route('/api/videos')
def api_videos():
    """Get all processed videos with their tags."""
    db = get_db()
    try:
        # Get all video tags from database
        rows = db.conn.execute("""
            SELECT video_path, content_tags, person_tags, processed_at
            FROM video_tags
            ORDER BY processed_at DESC
        """).fetchall()

        videos = []
        for row in rows:
            video_path = row[0]
            # Check if file still exists
            if os.path.exists(video_path):
                content_tags = row[1].split(",") if row[1] else []
                person_tags = row[2].split(",") if row[2] else []
                videos.append({
                    'path': video_path,
                    'filename': os.path.basename(video_path),
                    'content_tags': [t for t in content_tags if t],
                    'person_tags': [t for t in person_tags if t],
                    'processed_at': row[3]
                })

        return jsonify(videos)
    finally:
        db.close()


@app.route('/api/persons')
def api_persons():
    """Get all known persons."""
    db = get_db()
    try:
        persons = db.get_all_persons()
        return jsonify([{'id': p[0], 'name': p[1]} for p in persons])
    finally:
        db.close()


@app.route('/api/persons/<int:person_id>', methods=['PUT'])
def api_rename_person(person_id: int):
    """Rename a person."""
    db = get_db()
    try:
        data = request.get_json()
        new_name = data.get('name', '').strip()
        if not new_name:
            return jsonify({'error': 'Name is required'}), 400

        # Get old name for updating video tags
        old_person = db.conn.execute(
            "SELECT name FROM persons WHERE id = ?", (person_id,)
        ).fetchone()

        if not old_person:
            return jsonify({'error': 'Person not found'}), 404

        old_name = old_person[0]

        # Rename the person
        db.rename_person(person_id, new_name)

        # Update all video tags that reference this person
        rows = db.conn.execute(
            "SELECT video_path, person_tags FROM video_tags WHERE person_tags LIKE ?",
            (f"%{old_name}%",)
        ).fetchall()

        for row in rows:
            video_path = row[0]
            person_tags = row[1].split(",") if row[1] else []
            # Replace old name with new name
            updated_tags = [new_name if t == old_name else t for t in person_tags]
            db.conn.execute(
                "UPDATE video_tags SET person_tags = ? WHERE video_path = ?",
                (",".join(updated_tags), video_path)
            )
        db.conn.commit()

        return jsonify({'success': True, 'old_name': old_name, 'new_name': new_name})
    finally:
        db.close()


@app.route('/api/video')
def api_serve_video():
    """Serve a video file."""
    video_path = request.args.get('path')
    if not video_path:
        abort(400)

    video_path = unquote(video_path)
    if not os.path.exists(video_path):
        abort(404)

    # Security: ensure it's a video file
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in {e.lower() for e in VIDEO_EXTENSIONS}:
        abort(403)

    # Map extensions to MIME types
    mime_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
        '.m4v': 'video/x-m4v',
    }
    mimetype = mime_types.get(ext, 'video/mp4')

    return send_file(video_path, mimetype=mimetype)


@app.route('/video/<path:video_path>')
def video_page(video_path: str):
    """Page for viewing a single video with its tags."""
    video_path = '/' + unquote(video_path)
    return render_template('video.html', video_path=video_path)


def run_server(video_dir: str | None = None, host: str = '127.0.0.1', port: int = 5000):
    """Run the web server."""
    global VIDEO_DIR
    if video_dir:
        VIDEO_DIR = Path(video_dir)

    print(f"\n🎬 Video Tagger Web UI")
    print(f"   Open http://{host}:{port} in your browser\n")
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Video Tagger Web UI')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--dir', help='Video directory (optional)')
    args = parser.parse_args()

    run_server(args.dir, args.host, args.port)
