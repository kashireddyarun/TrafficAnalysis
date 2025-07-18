# Sample Videos Directory

This directory contains sample traffic videos for testing and demonstration purposes.

## Video Sources

You can add traffic videos here for testing the system. Recommended video formats:

- MP4, AVI, MOV
- Resolution: 720p or higher
- Frame rate: 30 FPS or higher

## Sample Video Downloads

Here are some sources where you can download sample traffic videos:

### Free Traffic Video Sources:

1. **Pixabay**: https://pixabay.com/videos/search/traffic/
2. **Pexels**: https://www.pexels.com/search/videos/traffic/
3. **Unsplash**: https://unsplash.com/s/videos/traffic
4. **YouTube** (with proper licensing): Search for "traffic intersection," "highway traffic," etc.

### Test Video Scenarios:

- **Light Traffic**: 5-10 vehicles per frame
- **Medium Traffic**: 15-25 vehicles per frame
- **Heavy Traffic**: 30+ vehicles per frame
- **Different Times**: Morning, afternoon, evening, night
- **Weather Conditions**: Clear, rainy, foggy
- **Camera Angles**: Overhead, side view, intersection view

## Usage

Place your video files in this directory and reference them in your processing commands:

```bash
# Process a sample video
python main.py --source "data/sample_videos/traffic_sample.mp4" --output "output/processed_traffic.mp4"

# Batch process all videos
python src/app/real_time_processor.py --batch data/sample_videos/*.mp4 --output-dir output/
```

## Video Naming Convention

Use descriptive names for better organization:

- `highway_light_traffic_day.mp4`
- `intersection_heavy_traffic_evening.mp4`
- `urban_medium_traffic_rain.mp4`
- `freeway_congested_rush_hour.mp4`

This helps in organizing test scenarios and understanding performance in different conditions.
