#!/usr/bin/env node

import { YoutubeTranscript } from 'youtube-transcript-plus';

const videoId = process.argv[2];

if (!videoId) {
  console.error('Usage: transcript.js <video-id-or-url>');
  console.error('Example: transcript.js EBw7gsDPAYQ');
  console.error('Example: transcript.js https://www.youtube.com/watch?v=EBw7gsDPAYQ');
  process.exit(1);
}

// Extract video ID if full URL is provided
let extractedId = videoId;
if (videoId.includes('youtube.com') || videoId.includes('youtu.be')) {
  const match = videoId.match(/(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})/);
  if (match) {
    extractedId = match[1];
  }
}

// Decode HTML entities (YouTube returns double-encoded entities like &amp;#39;)
function decodeHtmlEntities(text) {
  return text
    .replace(/&amp;/g, '&')      // First decode &amp; -> &
    .replace(/&#39;/g, "'")      // Then decode &#39; -> '
    .replace(/&quot;/g, '"')     // &quot; -> "
    .replace(/&lt;/g, '<')       // &lt; -> <
    .replace(/&gt;/g, '>')       // &gt; -> >
    .replace(/&nbsp;/g, ' ')     // &nbsp; -> space
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(code)); // numeric entities
}

try {
  const transcript = await YoutubeTranscript.fetchTranscript(extractedId);
  
  for (const entry of transcript) {
    // offset is already in seconds, not milliseconds
    const timestamp = formatTimestamp(entry.offset);
    const text = decodeHtmlEntities(entry.text);
    console.log(`[${timestamp}] ${text}`);
  }
} catch (error) {
  console.error('Error fetching transcript:', error.message);
  process.exit(1);
}

function formatTimestamp(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  return `${m}:${s.toString().padStart(2, '0')}`;
}
