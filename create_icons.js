import fs from 'fs';
import path from 'path';

// Valid 64x64 indigo PNG base64
const pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAZSURBVHhe7cExAQAAAMKg9U9tDQ8gAAAAAACABwyqAAFlLg+tAAAAAElFTkSuQmCC";

const buffer = Buffer.from(pngBase64, 'base64');
const iconsDir = path.join(process.cwd(), 'src-tauri', 'icons');

if (!fs.existsSync(iconsDir)) {
  fs.mkdirSync(iconsDir, { recursive: true });
}

fs.writeFileSync(path.join(process.cwd(), 'icon.png'), buffer);
fs.writeFileSync(path.join(iconsDir, '32x32.png'), buffer);
fs.writeFileSync(path.join(iconsDir, '128x128.png'), buffer);
fs.writeFileSync(path.join(iconsDir, '128x128@2x.png'), buffer);
fs.writeFileSync(path.join(iconsDir, 'icon.png'), buffer);
// Simple fallback ico copy
fs.writeFileSync(path.join(iconsDir, 'icon.ico'), buffer);
fs.writeFileSync(path.join(iconsDir, 'icon.icns'), buffer);

console.log("✓ Created placeholder icons in src-tauri/icons/");
