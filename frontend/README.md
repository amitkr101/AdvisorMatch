# AdvisorMatch Frontend

Modern web interface for the AdvisorMatch semantic search system.

## Quick Start

### 1. Start the Backend API

```bash
cd app
python3 api.py
```

The API should be running at: http://localhost:8000

### 2. Open the Frontend

Simply open `index.html` in your web browser:

```bash
cd frontend
open index.html  # macOS
# or
xdg-open index.html  # Linux
# or just double-click index.html in your file explorer
```

Alternatively, use a simple HTTP server:

```bash
# Python 3
python3 -m http.server 3000

# Then visit: http://localhost:3000
```

## Features

### Search Interface
- Clean, modern design with Texas A&M colors
- Real-time search with loading indicators
- Configurable number of results (1-20)
- Error handling with helpful messages

### Results Display
- Ranked professor cards with detailed information
- Score breakdown (similarity, recency, activity)
- Top matching publications for each professor
- Direct links to faculty profiles

### Responsive Design
- Mobile-friendly layout
- Smooth animations and transitions
- Professional styling

## Usage

1. Enter your research interests in the search box
   - Example: "machine learning for robotics"
   - Example: "natural language processing"
   - Example: "computer vision deep learning"

2. Click "Search" or press Enter

3. View ranked results with:
   - Professor name and department
   - Overall match score
   - Score breakdown (similarity, recency, activity)
   - Top matching publications
   - Links to faculty profiles

## Architecture

```
frontend/
├── index.html          # Main HTML page
├── css/
│   └── styles.css      # Styling and layout
└── js/
    └── app.js          # Application logic
```

## Customization

### Change API URL
Edit `js/app.js`:
```javascript
const API_BASE_URL = 'http://your-api-url:8000';
```

### Adjust Styling
Edit `css/styles.css` to modify:
- Colors (Texas A&M maroon and gold)
- Layout and spacing
- Animations and transitions

## Browser Compatibility

Works in all modern browsers:
- Chrome/Edge (recommended)
- Firefox
- Safari

## Troubleshooting

### "Search failed" error
- Make sure the backend API is running
- Check that API is at http://localhost:8000
- Verify CORS is enabled in the API

### No results found
- Try different search terms
- Use more general keywords
- Check that database has publications

### Styling issues
- Clear browser cache
- Make sure CSS file is loaded
- Check browser console for errors
