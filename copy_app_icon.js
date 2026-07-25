import fs from 'fs';
import path from 'path';

const iconSrc = path.join(process.cwd(), 'src-tauri', 'icons', 'icon.png');
const publicDir = path.join(process.cwd(), 'public');

if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

if (fs.existsSync(iconSrc)) {
  fs.copyFileSync(iconSrc, path.join(publicDir, 'app_icon.png'));
  console.log("✓ Copied icon.png to public/app_icon.png");
} else {
  console.log("src-tauri/icons/icon.png not found!");
}
