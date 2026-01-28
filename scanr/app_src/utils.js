import "./lib/CSInterface";

const csInterface = new window.CSInterface();
const path = csInterface.getSystemPath(window.SystemPath.EXTENSION);
const storagePath = path + "/storage";
const aiTmpDir = path + "/ai_tmp";
const AI_SERVER_URL = "http://localhost:8765";

let locale = {};

const openUrl = window.cep.util.openURLInDefaultBrowser;

const checkUpdate = async (currentVersion) => {
  try {
    const response = await fetch(
      "https://api.github.com/repos/ScanR/TypeR/releases",
      { headers: { Accept: "application/vnd.github.v3.html+json" } }
    );
    if (!response.ok) return null;
    const releases = await response.json();

    const parseVersion = (version) => {
      const cleanVersion = version.replace(/^v/, '');
      return cleanVersion.split('.').map(num => parseInt(num, 10));
    };

    const compareVersions = (v1, v2) => {
      const version1 = parseVersion(v1);
      const version2 = parseVersion(v2);

      for (let i = 0; i < Math.max(version1.length, version2.length); i++) {
        const num1 = version1[i] || 0;
        const num2 = version2[i] || 0;

        if (num1 > num2) return 1;
        if (num1 < num2) return -1;
      }
      return 0;
    };

    const currentVersionClean = currentVersion.replace(/^v/, '');
    const newerReleases = releases.filter(release => {
      const releaseVersion = release.tag_name.replace(/^v/, '');
      return compareVersions(releaseVersion, currentVersionClean) > 0;
    });

    if (newerReleases.length > 0) {
      newerReleases.sort((a, b) => compareVersions(b.tag_name, a.tag_name));

      // Get the download URL for the latest release ZIP
      const latestRelease = newerReleases[0];
      let downloadUrl = null;

      // Try to find TypeR.zip in assets first
      if (latestRelease.assets && latestRelease.assets.length > 0) {
        const zipAsset = latestRelease.assets.find(a =>
          a.name.toLowerCase().endsWith('.zip') &&
          a.name.toLowerCase().includes('typer')
        );
        if (zipAsset) {
          downloadUrl = zipAsset.browser_download_url;
        }
      }
      // Fallback to zipball_url (source code zip)
      if (!downloadUrl) {
        downloadUrl = latestRelease.zipball_url;
      }

      return {
        version: newerReleases[0].tag_name,
        downloadUrl: downloadUrl,
        releases: newerReleases.map(release => ({
          version: release.tag_name,
          body: release.body_html || release.body,
          published_at: release.published_at
        }))
      };
    }
  } catch (e) {
    console.error("Update check failed", e);
  }
  return null;
};

const getOSType = () => {
  const os = csInterface.getOSInformation();
  if (os && os.toLowerCase().indexOf('mac') !== -1) {
    return 'mac';
  }
  return 'win';
};

const downloadAndInstallUpdate = async (downloadUrl, onProgress, onComplete, onError) => {
  try {
    const osType = getOSType();

    // Get user's Downloads folder
    const userHome = osType === 'win'
      ? csInterface.getSystemPath(window.SystemPath.USER_DATA).split('/AppData/')[0]
      : csInterface.getSystemPath(window.SystemPath.USER_DATA).replace('/Library/Application Support', '');

    const downloadsPath = osType === 'win'
      ? `${userHome}/Downloads/TypeR_Update`
      : `${userHome}/Downloads/TypeR_Update`;

    const zipPath = `${downloadsPath}/TypeR.zip`;

    onProgress && onProgress(locale.updateDownloading || 'Downloading update...');

    // Clean and create download directory
    csInterface.evalScript(`deleteFolder("${downloadsPath.replace(/\\/g, '\\\\').replace(/\//g, '\\\\')}")`, () => {
      // Use cep.fs to create directory
      const mkdirResult = window.cep.fs.makedir(downloadsPath);
      if (mkdirResult.err && mkdirResult.err !== 0 && mkdirResult.err !== 17) { // 17 = already exists
        onError && onError('Failed to create download directory');
        return;
      }

      // Download the ZIP file
      fetch(downloadUrl, {
        headers: { Accept: 'application/octet-stream' }
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Download failed: ${response.status}`);
          }
          return response.arrayBuffer();
        })
        .then(arrayBuffer => {
          const uint8Array = new Uint8Array(arrayBuffer);

          // Convert to base64 for file writing
          let binary = '';
          const len = uint8Array.byteLength;
          for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(uint8Array[i]);
          }
          const base64Data = window.btoa(binary);

          onProgress && onProgress(locale.updateExtracting || 'Extracting files...');

          // Write ZIP file using base64 encoding
          const writeResult = window.cep.fs.writeFile(zipPath, base64Data, window.cep.encoding.Base64);
          if (writeResult.err) {
            throw new Error('Failed to write ZIP file');
          }

          // Create the auto-install script
          if (osType === 'win') {
            // Windows: Create PowerShell install script
            const installScript = `# TypeR Auto-Update Script
# This script will install the update after Photoshop is closed
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$zipPath = Join-Path $ScriptDir "TypeR.zip"
$extractPath = Join-Path $ScriptDir "extracted"
$AppData = $env:APPDATA
$TargetDir = Join-Path $AppData "Adobe\\CEP\\extensions\\typertools"
$TempBackupContainer = Join-Path $env:TEMP "typer_backup_container"

Write-Host "+------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host "|                      TypeR Auto-Updater                          |" -ForegroundColor Cyan
Write-Host "+------------------------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""

# Check if Photoshop is running
$psProcess = Get-Process -Name "Photoshop" -ErrorAction SilentlyContinue
if ($psProcess) {
    Write-Host "[!] Photoshop is running. Please close it first." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter after closing Photoshop..."
}

Write-Host "[*] Installing update..." -ForegroundColor Cyan

# Cleanup temp backup
if (Test-Path $TempBackupContainer) { Remove-Item $TempBackupContainer -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -Path $TempBackupContainer -ItemType Directory -Force | Out-Null

# Backup storage
if (Test-Path "$TargetDir\\storage") {
    Copy-Item "$TargetDir\\storage" -Destination $TempBackupContainer -Recurse -Force -ErrorAction SilentlyContinue
}

# Extract ZIP
if (Test-Path $extractPath) { Remove-Item $extractPath -Recurse -Force }
New-Item -Path $extractPath -ItemType Directory -Force | Out-Null
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

# Find content folder - check if files are at root or in a subfolder
# If CSXS folder exists at root, files are directly there
# Otherwise, look for a subfolder containing CSXS
if (Test-Path "$extractPath\\CSXS") {
    $sourcePath = $extractPath
} else {
    $contentFolder = Get-ChildItem -Path $extractPath -Directory | Where-Object { Test-Path "$($_.FullName)\\CSXS" } | Select-Object -First 1
    if ($contentFolder) {
        $sourcePath = $contentFolder.FullName
    } else {
        $sourcePath = $extractPath
    }
}

# Clean target directory
if (Test-Path $TargetDir) {
    Remove-Item $TargetDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -Path $TargetDir -ItemType Directory -Force | Out-Null

# Copy files
$FoldersToCopy = @("app", "CSXS", "icons", "locale")
foreach ($folder in $FoldersToCopy) {
    $src = Join-Path $sourcePath $folder
    $dst = Join-Path $TargetDir $folder
    if (Test-Path $src) {
        Copy-Item $src -Destination $dst -Recurse -Force
    }
}

# Copy themes
if (Test-Path "$sourcePath\\themes") {
    $ThemeDest = "$TargetDir\\app\\themes"
    if (-not (Test-Path $ThemeDest)) { New-Item $ThemeDest -ItemType Directory -Force | Out-Null }
    Copy-Item "$sourcePath\\themes\\*" -Destination $ThemeDest -Recurse -Force
}

# Restore storage
if (Test-Path "$TempBackupContainer\\storage") {
    Copy-Item "$TempBackupContainer\\storage" -Destination "$TargetDir" -Recurse -Force
}

# Cleanup
if (Test-Path $TempBackupContainer) { Remove-Item $TempBackupContainer -Recurse -Force -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "+------------------------------------------------------------------+" -ForegroundColor Green
Write-Host "|                      Update Complete!                            |" -ForegroundColor Green
Write-Host "+------------------------------------------------------------------+" -ForegroundColor Green
Write-Host ""
Write-Host "You can now open Photoshop and use TypeR." -ForegroundColor Cyan
Write-Host ""
Write-Host "This folder will be deleted automatically." -ForegroundColor DarkGray
Read-Host "Press Enter to exit..."

# Cleanup update folder - delete the entire TypeR_Update folder
$parentDir = Split-Path $ScriptDir -Parent
$folderName = Split-Path $ScriptDir -Leaf
Set-Location $parentDir
Remove-Item $ScriptDir -Recurse -Force -ErrorAction SilentlyContinue
`;

            const cmdScript = `@echo off
cd /d "%~dp0"
PowerShell -NoProfile -ExecutionPolicy Bypass -File "install_update.ps1"
`;

            const psScriptPath = `${downloadsPath}/install_update.ps1`;
            const cmdScriptPath = `${downloadsPath}/install_update.cmd`;

            window.cep.fs.writeFile(psScriptPath, installScript);
            window.cep.fs.writeFile(cmdScriptPath, cmdScript);

            onProgress && onProgress(locale.updateReady || 'Update ready to install...');

            // Open the folder in Explorer
            csInterface.evalScript(`openFolder("${downloadsPath.replace(/\\/g, '\\\\').replace(/\//g, '\\\\')}")`, () => {
              onComplete && onComplete(true); // true = needs manual step
            });

          } else {
            // macOS: Create shell install script
            const installScript = `#!/bin/bash
# TypeR Auto-Update Script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_PATH="$SCRIPT_DIR/TypeR.zip"
EXTRACT_PATH="$SCRIPT_DIR/extracted"
DEST_DIR="$HOME/Library/Application Support/Adobe/CEP/extensions/typertools"
TEMP_STORAGE="$SCRIPT_DIR/__storage_backup"

echo "+------------------------------------------------------------------+"
echo "|                      TypeR Auto-Updater                          |"
echo "+------------------------------------------------------------------+"
echo ""

# Check if Photoshop is running
if pgrep -x "Adobe Photoshop" > /dev/null; then
    echo "[!] Photoshop is running. Please close it first."
    echo ""
    read -p "Press Enter after closing Photoshop..."
fi

echo "[*] Installing update..."

# Backup storage
if [ -e "$DEST_DIR/storage" ]; then
    cp "$DEST_DIR/storage" "$TEMP_STORAGE"
fi

# Extract ZIP
rm -rf "$EXTRACT_PATH"
mkdir -p "$EXTRACT_PATH"
unzip -o "$ZIP_PATH" -d "$EXTRACT_PATH"

# Find content folder - check if files are at root or in a subfolder
if [ -d "$EXTRACT_PATH/CSXS" ]; then
    SOURCE_PATH="$EXTRACT_PATH"
else
    CONTENT_FOLDER=$(find "$EXTRACT_PATH" -maxdepth 2 -type d -name "CSXS" | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$CONTENT_FOLDER" ]; then
        SOURCE_PATH="$CONTENT_FOLDER"
    else
        SOURCE_PATH="$EXTRACT_PATH"
    fi
fi

# Clean and recreate target
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# Copy files
for folder in app CSXS icons locale; do
    if [ -d "$SOURCE_PATH/$folder" ]; then
        cp -r "$SOURCE_PATH/$folder" "$DEST_DIR/"
    fi
done

# Copy themes
if [ -d "$SOURCE_PATH/themes" ]; then
    mkdir -p "$DEST_DIR/app/themes"
    cp -r "$SOURCE_PATH/themes/"* "$DEST_DIR/app/themes/"
fi

# Restore storage
if [ -f "$TEMP_STORAGE" ]; then
    cp "$TEMP_STORAGE" "$DEST_DIR/storage"
fi

echo ""
echo "+------------------------------------------------------------------+"
echo "|                      Update Complete!                            |"
echo "+------------------------------------------------------------------+"
echo ""
echo "You can now open Photoshop and use TypeR."
echo ""
echo "This folder will be deleted automatically."
read -p "Press Enter to exit..."

# Cleanup - delete the entire TypeR_Update folder
cd "$HOME/Downloads"
rm -rf "$SCRIPT_DIR"
`;

            const shScriptPath = `${downloadsPath}/install_update.command`;
            window.cep.fs.writeFile(shScriptPath, installScript);

            // Make executable
            csInterface.evalScript(`makeExecutable("${shScriptPath}")`, () => {
              onProgress && onProgress(locale.updateReady || 'Update ready to install...');

              // Open the folder in Finder
              csInterface.evalScript(`openFolder("${downloadsPath}")`, () => {
                onComplete && onComplete(true); // true = needs manual step
              });
            });
          }
        })
        .catch(err => {
          console.error('Update failed:', err);
          onError && onError(err.message || 'Update failed');
        });
    });

  } catch (e) {
    console.error('Update failed:', e);
    onError && onError(e.message || 'Update failed');
  }
};

const readStorage = (key) => {
  const result = window.cep.fs.readFile(storagePath);
  if (result.err) {
    return key
      ? void 0
      : {
        error: result.err,
        data: {},
      };
  } else {
    const data = JSON.parse(result.data || "{}") || {};
    return key ? data[key] : { data };
  }
};

const writeToStorage = (data, rewrite) => {
  const storage = readStorage();
  if (storage.error || rewrite) {
    const result = window.cep.fs.writeFile(storagePath, JSON.stringify(data));
    return !result.err;
  } else {
    data = Object.assign({}, storage.data, data);
    const result = window.cep.fs.writeFile(storagePath, JSON.stringify(data));
    return !result.err;
  }
};

const deleteStorageFile = () => {
  const result = window.cep.fs.deleteFile(storagePath);
  if (typeof result === "number") {
    return (
      result === window.cep.fs.NO_ERROR ||
      result === window.cep.fs.ERR_NOT_FOUND
    );
  }
  if (typeof result === "object" && result) {
    return !result.err || result.err === window.cep.fs.ERR_NOT_FOUND;
  }
  return false;
};

const parseLocaleFile = (str) => {
  const result = {};
  if (!str) return result;
  const lines = str.replace(/\r/g, "").split("\n");
  let key = null;
  let val = "";
  for (let line of lines) {
    if (line.startsWith("#")) continue;
    if (key) {
      val += line;
      if (val.endsWith("\\")) {
        val = val.slice(0, -1) + "\n";
        continue;
      }
      result[key] = val;
      key = null;
      val = "";
      continue;
    }
    const i = line.indexOf("=");
    if (i === -1) continue;
    key = line.slice(0, i).trim();
    val = line.slice(i + 1);
    if (val.endsWith("\\")) {
      val = val.slice(0, -1) + "\n";
      continue;
    }
    result[key] = val;
    key = null;
    val = "";
  }
  return result;
};

const initLocale = () => {
  locale = csInterface.initResourceBundle();
  const loadLocaleFile = (file) => {
    const result = window.cep.fs.readFile(file);
    if (!result.err) {
      const data = parseLocaleFile(result.data);
      locale = Object.assign(locale, data);
    }
  };
  // Always merge default strings to ensure fallbacks for new keys
  loadLocaleFile(`${path}/locale/messages.properties`);
  const lang = readStorage("language");
  if (lang && lang !== "auto") {
    const file = lang === "en_US" ? `${path}/locale/messages.properties` : `${path}/locale/${lang}/messages.properties`;
    loadLocaleFile(file);
  }
};

initLocale();

const nativeAlert = (text, title, isError) => {
  const data = JSON.stringify({ text, title, isError });
  csInterface.evalScript("nativeAlert(" + data + ")");
};

const nativeConfirm = (text, title, callback) => {
  const data = JSON.stringify({ text, title });
  csInterface.evalScript("nativeConfirm(" + data + ")", (result) => callback(!!result));
};

const _safeParseJSON = (raw) => {
  try {
    return JSON.parse(raw || "{}");
  } catch (e) {
    return {};
  }
};

const _evalScriptJSON = (script) => {
  return new Promise((resolve) => {
    csInterface.evalScript(script, (result) => resolve(_safeParseJSON(result)));
  });
};

let userFonts = null;
const getUserFonts = () => {
  return Array.isArray(userFonts) ? userFonts.concat([]) : [];
};
if (!userFonts) {
  csInterface.evalScript("getUserFonts()", (data) => {
    const dataObj = JSON.parse(data || "{}");
    const fonts = dataObj.fonts || [];
    userFonts = fonts;
  });
}

const getActiveLayerText = (callback) => {
  csInterface.evalScript("getActiveLayerText()", (data) => {
    const dataObj = JSON.parse(data || "{}");
    if (!data || !dataObj.textProps) nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
    else callback(dataObj);
  });
};

const setActiveLayerText = (text, style, direction, callback = () => { }) => {
  // Support legacy calls where direction is omitted and callback is 3rd parameter
  if (typeof direction === "function") {
    callback = direction;
    direction = undefined;
  }
  if (!text && !style) {
    nativeAlert(locale.errorNoTextNoStyle, locale.errorTitle, true);
    callback(false);
    return false;
  }
  const data = JSON.stringify({ text, style, direction });
  csInterface.evalScript("setActiveLayerText(" + data + ")", (error) => {
    if (error) nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
    callback(!error);
  });
};

const getCurrentSelection = (callback = () => { }) => {
  csInterface.evalScript("getCurrentSelection()", (result) => {
    const data = JSON.parse(result || "{}");
    if (data.error) {
      callback(null);
    } else {
      callback(data);
    }
  });
};

const getSelectionBoundsHash = (selection) => {
  if (!selection) return null;
  return `${selection.xMid}_${selection.yMid}_${selection.width}_${selection.height}`;
};

const startSelectionMonitoring = () => {
  csInterface.evalScript("startSelectionMonitoring()");
};

const stopSelectionMonitoring = () => {
  csInterface.evalScript("stopSelectionMonitoring()");
};

const getSelectionChanged = (callback = () => { }) => {
  csInterface.evalScript("getSelectionChanged()", (result) => {
    const data = JSON.parse(result || "{}");
    if (data.noChange) {
      callback(null);
    } else if (data.error) {
      callback(null);
    } else {
      callback(data);
    }
  });
};

const createTextLayerInSelection = (text, style, pointText, padding, direction, callback = () => { }) => {
  // Support legacy calls where padding/direction are omitted and callback may be 4th or 5th parameter
  if (typeof padding === "function") {
    callback = padding;
    padding = 0;
    direction = undefined;
  } else if (typeof direction === "function") {
    callback = direction;
    direction = undefined;
  }
  if (!text) {
    nativeAlert(locale.errorNoText, locale.errorTitle, true);
    callback(false);
    return false;
  }
  if (!style) {
    style = { textProps: getDefaultStyle(), stroke: getDefaultStroke() };
  }
  const data = JSON.stringify({ text, style, padding: padding || 0, direction });
  csInterface.evalScript("createTextLayerInSelection(" + data + ", " + !!pointText + ")", (error) => {
    if (error === "smallSelection") nativeAlert(locale.errorSmallSelection, locale.errorTitle, true);
    else if (error) nativeAlert(locale.errorNoSelection, locale.errorTitle, true);
    callback(!error);
  });
};

const createTextLayersInStoredSelections = (texts, styles, selections, pointText, padding, direction, callback = () => { }) => {
  // Support legacy calls where padding/direction are omitted and callback may be 5th or 6th parameter
  if (typeof padding === "function") {
    callback = padding;
    padding = 0;
    direction = undefined;
  } else if (typeof direction === "function") {
    callback = direction;
    direction = undefined;
  }
  if (!Array.isArray(texts) || texts.length === 0) {
    nativeAlert(locale.errorNoText, locale.errorTitle, true);
    callback(false);
    return false;
  }
  if (!Array.isArray(styles) || styles.length === 0) {
    styles = [{ textProps: getDefaultStyle(), stroke: getDefaultStroke() }];
  }
  if (!Array.isArray(selections) || selections.length === 0) {
    nativeAlert(locale.errorNoSelection, locale.errorTitle, true);
    callback(false);
    return false;
  }
  const data = JSON.stringify({ texts, styles, selections, padding: padding || 0, direction });
  csInterface.evalScript("createTextLayersInStoredSelections(" + data + ", " + !!pointText + ")", (error) => {
    if (error === "smallSelection") nativeAlert(locale.errorSmallSelection, locale.errorTitle, true);
    else if (error === "noSelection") nativeAlert(locale.errorNoSelection, locale.errorTitle, true);
    else if (error) nativeAlert("Error: " + error, locale.errorTitle, true);
    callback(!error);
  });
};

const alignTextLayerToSelection = (resizeTextBox = false, padding = 0) => {
  const data = JSON.stringify({ resizeTextBox: !!resizeTextBox, padding: padding || 0 });
  csInterface.evalScript("alignTextLayerToSelection(" + data + ")", (error) => {
    if (error === "smallSelection") nativeAlert(locale.errorSmallSelection, locale.errorTitle, true);
    else if (error === "noSelection") nativeAlert(locale.errorNoSelection, locale.errorTitle, true);
    else if (error) nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
  });
};

const alignTextLayerToTarget = (target, resizeTextBox = false, padding = 0) => {
  const data = JSON.stringify({ target, resizeTextBox: !!resizeTextBox, padding: padding || 0 });
  csInterface.evalScript("alignTextLayerToTarget(" + data + ")", (error) => {
    if (error === "target") nativeAlert(locale.errorAiInvalidTarget || "AI error: invalid target.", locale.errorTitle, true);
    else if (error === "layer") nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
    else if (error) nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
  });
};

const aiLoadBubbleModel = async () => {
  const response = await fetch(`${AI_SERVER_URL}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail ? `\n\n${data.detail}` : "";
    throw new Error((data?.error || "AI server error") + detail);
  }
  return data;
};

const aiUnloadBubbleModel = async () => {
  const response = await fetch(`${AI_SERVER_URL}/unload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail ? `\n\n${data.detail}` : "";
    throw new Error((data?.error || "AI server error") + detail);
  }
  return data;
};

const _ensureAiTmpDir = () => {
  const result = window.cep.fs.makedir(aiTmpDir);
  // 17 = already exists (CEP), so ignore.
  return !result.err || result.err === 17;
};

const _exportAiRoi = async (roi) => {
  const res = await _evalScriptJSON(`exportAiRoi(${JSON.stringify(roi)})`);
  if (!res || res.error) {
    const err = res?.error || "export";
    throw new Error(`AI ROI export failed: ${err}`);
  }
  return res;
};

const _readFileBase64 = (filePath) => {
  const result = window.cep.fs.readFile(filePath, window.cep.encoding.Base64);
  if (result.err) throw new Error(`Failed to read ROI file: ${result.err}`);
  return result.data;
};

const _deleteFileSafe = (filePath) => {
  try {
    window.cep.fs.deleteFile(filePath);
  } catch (e) { }
};

const exportActiveDocFlattenPng = async (prefix = "autotranslate") => {
  _ensureAiTmpDir();
  const outPath = `${aiTmpDir}/${prefix}_${Date.now()}.png`;
  const res = await _evalScriptJSON(`exportActiveDocPng(${JSON.stringify({ path: outPath })})`);
  if (!res || res.error) {
    throw new Error(res?.detail || res?.error || "export");
  }
  return res;
};

const createBubbleRectanglesGroup = async ({
  groupName = "BUBBLES_DETECTED",
  bubbles = [],
  strokePx = 4,
  color = { r: 0, g: 120, b: 255 },
  fillOpacity = 0,
} = {}) => {
  const payload = {
    groupName,
    bubbles,
    stroke: { size: strokePx, opacity: 100, enabled: true, color },
    fillOpacity,
  };
  const res = await _evalScriptJSON(`createBubbleRectangles(${JSON.stringify(payload)})`);
  if (!res || res.error) {
    throw new Error(res?.detail || res?.error || "createBubbleRectangles");
  }
  return res;
};

const readBubbleRectanglesGroup = async (groupName = "BUBBLES_DETECTED") => {
  const res = await _evalScriptJSON(`getRectanglesFromGroup(${JSON.stringify({ groupName })})`);
  if (!res || res.error) {
    throw new Error(res?.detail || res?.error || "getRectanglesFromGroup");
  }
  return res.rectangles || [];
};

const renumberBubbleRectanglesGroup = async (groupName = "BUBBLES_DETECTED", rowToleranceFactor = 0.5, readingOrder) => {
  const payload = { groupName, rowToleranceFactor };
  if (readingOrder) payload.readingOrder = readingOrder;
  const res = await _evalScriptJSON(`renumberBubbleGroup(${JSON.stringify(payload)})`);
  if (!res || res.error) {
    throw new Error(res?.detail || res?.error || "renumberBubbleGroup");
  }
  return res.bubbles || [];
};

const createTranslatedTextLayers = async ({
  groupName = "TRANSLATION",
  bubbles = [],
  translations = [],
  style = null,
  fit = { minSize: 6, maxSize: null, padding: 10 },
  styles = null,
  direction,
} = {}) => {
  const basePayload = { groupName, bubbles, translations, style, fit, styles, direction };
  const baseJson = JSON.stringify(basePayload);

  // Avoid passing huge payloads through evalScript (can destabilize CEP/Photoshop on some systems).
  const MAX_EVALSCRIPT_CHARS = 350000;
  const MAX_BUBBLES_PER_CHUNK = 80;
  if (bubbles.length <= MAX_BUBBLES_PER_CHUNK && baseJson.length <= MAX_EVALSCRIPT_CHARS) {
    const res = await _evalScriptJSON(`createTranslatedTextLayers(${baseJson})`);
    if (!res || res.error) {
      throw new Error(res?.detail || res?.error || "createTranslatedTextLayers");
    }
    return res;
  }

  const translationMap = new Map();
  for (const t of translations || []) {
    if (t && t.id) {
      translationMap.set(t.id, t.text || "");
    }
  }

  const fullFitMap = fit && fit.map && typeof fit.map === "object" && !Array.isArray(fit.map) ? fit.map : null;
  const fullStyles = styles && typeof styles === "object" && !Array.isArray(styles) ? styles : null;

  let totalCreated = 0;
  for (let i = 0; i < (bubbles || []).length; i += MAX_BUBBLES_PER_CHUNK) {
    const chunkBubbles = bubbles.slice(i, i + MAX_BUBBLES_PER_CHUNK);
    const chunkIds = chunkBubbles.map((b) => b?.id).filter(Boolean);

    const chunkTranslations = chunkIds.map((id) => ({ id, text: translationMap.get(id) || "" }));

    let chunkFitMap = null;
    if (fullFitMap) {
      chunkFitMap = {};
      for (const id of chunkIds) {
        if (fullFitMap[id] != null) chunkFitMap[id] = fullFitMap[id];
      }
    }

    let chunkStyles = null;
    if (fullStyles) {
      chunkStyles = {};
      for (const id of chunkIds) {
        if (fullStyles[id] != null) chunkStyles[id] = fullStyles[id];
      }
    }

    const fitToSend = fullFitMap || chunkFitMap ? { ...(fit || {}), map: chunkFitMap || fullFitMap || null } : fit;
    const stylesToSend = fullStyles || chunkStyles ? chunkStyles || fullStyles || null : styles;

    const payload = {
      groupName,
      bubbles: chunkBubbles,
      translations: chunkTranslations,
      style,
      fit: fitToSend,
      styles: stylesToSend,
      direction,
      append: i > 0,
    };

    const res = await _evalScriptJSON(`createTranslatedTextLayers(${JSON.stringify(payload)})`);
    if (!res || res.error) {
      throw new Error(res?.detail || res?.error || "createTranslatedTextLayers");
    }
    totalCreated += Number(res.count) || chunkBubbles.length;
  }

  return { ok: true, count: totalCreated, groupName };
};

const applyCleanedPngToActiveDoc = async ({
  path,
  groupName = "CLEANED",
  belowGroupName = "BUBBLES_DETECTED",
  translationGroupName = "TRANSLATION",
} = {}) => {
  const payload = { path, groupName, belowGroupName, translationGroupName };
  const res = await _evalScriptJSON(`applyCleanedPng(${JSON.stringify(payload)})`);
  if (!res || res.error) {
    throw new Error(res?.detail || res?.error || "applyCleanedPng");
  }
  return res;
};

const openFolderPath = (folderPath) => {
  csInterface.evalScript(`openFolder(${JSON.stringify(folderPath || "")})`);
};

const TIPER_SERVER_URL = AI_SERVER_URL;

const _serverFetchJson = async (url, options) => {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data?.detail || data?.error || data;
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
    throw new Error(msg);
  }
  return data;
};

const serverCreateJob = async ({ sourcePngPath, config, signal } = {}) => {
  const data = await _serverFetchJson(`${TIPER_SERVER_URL}/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_png_path: sourcePngPath, config: config || undefined }),
    signal,
  });
  return data.job;
};

const serverStartDetectBubbles = async (jobId, { signal } = {}) => {
  return _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}/detect_bubbles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
    signal,
  });
};

const serverGetJob = async (jobId, { signal } = {}) => {
  const data = await _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}`, {
    method: "GET",
    signal,
  });
  return data.job;
};

const serverGetBubblesAuto = async (jobId, { signal } = {}) => {
  return _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}/bubbles_auto`, {
    method: "GET",
    signal,
  });
};

const serverSubmitBubbles = async (jobId, bubbles, { signal } = {}) => {
  return _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}/bubbles`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bubbles }),
    signal,
  });
};

const serverStartOcrClean = async (jobId, { signal } = {}) => {
  return _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}/run_ocr_clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
    signal,
  });
};

const serverGetTranslations = async (jobId, payload = {}, { signal } = {}) => {
  return _serverFetchJson(`${TIPER_SERVER_URL}/jobs/${encodeURIComponent(jobId)}/get_translation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
    signal,
  });
};

const alignTextLayerToBubbleAI = async (resizeTextBox = false, padding = 0) => {
  let docInfo = await _evalScriptJSON("getActiveDocumentInfo()");
  if (!docInfo || docInfo.error) {
    nativeAlert(locale.errorNoSelection || "No open document.", locale.errorTitle, true);
    return false;
  }

  let layerBounds = await _evalScriptJSON("getActiveLayerBounds()");
  if (!layerBounds || layerBounds.error) {
    nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
    return false;
  }

  // Get textbox/paragraph bounds for ROI size calculation
  let textBoxBounds = await _evalScriptJSON("getActiveTextBoxBounds()");
  if (!textBoxBounds || textBoxBounds.error) {
    // Fallback to layer bounds if textbox bounds unavailable
    textBoxBounds = layerBounds;
  }

  const point = { x: layerBounds.xMid, y: layerBounds.yMid };
  const textW = Math.max(1, textBoxBounds.width || 1);
  const textH = Math.max(1, textBoxBounds.height || 1);
  const basePad = Math.max(textW, textH) * 4;
  const padTries = [basePad, basePad * 1.5, basePad * 2.2];

  _ensureAiTmpDir();

  for (let attempt = 0; attempt < padTries.length; attempt++) {
    const pad = padTries[attempt];
    const left = Math.max(0, Math.floor(layerBounds.left - pad));
    const top = Math.max(0, Math.floor(layerBounds.top - pad));
    const right = Math.min(docInfo.width, Math.ceil(layerBounds.right + pad));
    const bottom = Math.min(docInfo.height, Math.ceil(layerBounds.bottom + pad));

    const roiPath = `${aiTmpDir}/roi_${Date.now()}_${attempt}.png`;

    let roiExport = null;
    try {
      roiExport = await _exportAiRoi({ path: roiPath, left, top, right, bottom });
      const imgBase64 = _readFileBase64(roiExport.path);

      const response = await fetch(`${AI_SERVER_URL}/detect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: imgBase64,
          roi_offset: { x: roiExport.left, y: roiExport.top },
          point,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data?.detail ? `\n\n${data.detail}` : "";
        throw new Error((data?.error || "AI server error") + detail);
      }

      if (!data.selected) {
        continue;
      }

      const core = data.selected.core_bbox || data.selected.bbox;
      const target = {
        xMid: data.selected.center.x,
        yMid: data.selected.center.y,
        width: Math.max(1, (core.x2 - core.x1) || 1),
        height: Math.max(1, (core.y2 - core.y1) || 1),
      };

      alignTextLayerToTarget(target, resizeTextBox, padding);
      return true;
    } catch (e) {
      console.error(e);
      // Retry with a larger ROI, unless it's clearly a server issue.
      if (attempt === padTries.length - 1) {
        nativeAlert(
          (locale.errorAiAlignFailed || "AI alignment failed.") +
          "\n\n" +
          (locale.errorAiServerHint || 'Run: python bubble_detector/server.py') +
          "\n\n" +
          (e && e.message ? e.message : ""),
          locale.errorTitle,
          true
        );
        return false;
      }
    } finally {
      if (roiExport && roiExport.path) {
        _deleteFileSafe(roiExport.path);
      } else {
        _deleteFileSafe(roiPath);
      }
    }
  }

  nativeAlert(locale.errorAiNoBubbleFound || "AI: bubble not found near the active text layer.", locale.errorTitle, true);
  return false;
};

/**
 * Fixes trailing spaces in the active text layer by replacing " " at the end of lines with "\r" (enter).
 * Returns a Promise that resolves to true if text was modified, false otherwise.
 */
const fixTrailingSpacesInActiveLayer = () => {
  return new Promise((resolve) => {
    csInterface.evalScript("getActiveLayerText()", (data) => {
      if (!data) {
        resolve(false);
        return;
      }
      const dataObj = JSON.parse(data || "{}");
      if (!dataObj.textProps || !dataObj.textProps.layerText || !dataObj.textProps.layerText.textKey) {
        resolve(false);
        return;
      }

      const originalText = dataObj.textProps.layerText.textKey;
      // Replace trailing space at end of lines (before \r or \n) with \r
      // Also handle lines ending with space followed by nothing (last line)
      const fixedText = originalText.replace(/ (?=\r|\n|$)/g, "\r");

      if (fixedText === originalText) {
        resolve(false);
        return;
      }

      // Set the fixed text back (keeping all other style properties)
      const payload = JSON.stringify({ text: fixedText, style: null, direction: undefined });
      csInterface.evalScript("setActiveLayerText(" + payload + ")", (error) => {
        resolve(!error);
      });
    });
  });
};

const changeActiveLayerTextSize = (val, callback = () => { }) => {
  csInterface.evalScript("changeActiveLayerTextSize(" + val + ")", (error) => {
    if (error) nativeAlert(locale.errorNoTextLayer, locale.errorTitle, true);
    callback(!error);
  });
};

const getHotkeyPressed = (callback) => {
  csInterface.evalScript("getHotkeyPressed()", callback);
};

const resizeTextArea = () => {
  const textArea = document.querySelector(".text-area");
  const textLines = document.querySelector(".text-lines");
  if (textArea && textLines) {
    textArea.style.height = textLines.offsetHeight + "px";
  }
};

const scrollToLine = (lineIndex, delay = 300) => {
  lineIndex = lineIndex < 5 ? 0 : lineIndex - 5;
  setTimeout(() => {
    const line = document.querySelectorAll(".text-line")[lineIndex];
    if (line) line.scrollIntoView();
  }, delay);
};

const scrollToStyle = (styleId, delay = 100) => {
  setTimeout(() => {
    const style = document.getElementById(styleId);
    if (style) style.scrollIntoView();
  }, delay);
};

const rgbToHex = (rgb = {}) => {
  const componentToHex = (c = 0) => ("0" + Math.round(c).toString(16)).substr(-2).toUpperCase();
  const r = rgb.red != null ? rgb.red : rgb.r;
  const g = rgb.green != null ? rgb.green : rgb.g;
  const b = rgb.blue != null ? rgb.blue : rgb.b;
  return "#" + componentToHex(r) + componentToHex(g) + componentToHex(b);
};

const getStyleObject = (textStyle) => {
  const styleObj = {};
  if (textStyle.fontName) styleObj.fontFamily = textStyle.fontName;
  if (textStyle.fontPostScriptName) styleObj.fontFileFamily = textStyle.fontPostScriptName;
  if (textStyle.syntheticBold) styleObj.fontWeight = "bold";
  if (textStyle.syntheticItalic) styleObj.fontStyle = "italic";
  if (textStyle.fontCaps === "allCaps") styleObj.textTransform = "uppercase";
  if (textStyle.fontCaps === "smallCaps") styleObj.textTransform = "lowercase";
  if (textStyle.underline && textStyle.underline !== "underlineOff") styleObj.textDecoration = "underline";
  if (textStyle.strikethrough && textStyle.strikethrough !== "strikethroughOff") {
    if (styleObj.textDecoration) styleObj.textDecoration += " line-through";
    else styleObj.textDecoration = "line-through";
  }
  return styleObj;
};

const getDefaultStyle = () => {
  return {
    layerText: {
      textGridding: "none",
      orientation: "horizontal",
      antiAlias: "antiAliasSmooth",
      textStyleRange: [
        {
          from: 0,
          to: 100,
          textStyle: {
            fontPostScriptName: "v_CCWildWordsRoman",
            fontName: "v_CCWild Words Roman",
            fontStyleName: "Regular",
            fontScript: 0,
            fontTechnology: 1,
            fontAvailable: true,
            size: 14,
            impliedFontSize: 14,
            horizontalScale: 100,
            verticalScale: 100,
            autoLeading: true,
            tracking: 0,
            baselineShift: 0,
            impliedBaselineShift: 0,
            autoKern: "metricsKern",
            fontCaps: "normal",
            digitSet: "defaultDigits",
            diacXOffset: 0,
            markYDistFromBaseline: 100,
            otbaseline: "normal",
            ligature: false,
            altligature: false,
            connectionForms: false,
            contextualLigatures: false,
            baselineDirection: "withStream",
            color: { red: 0, green: 0, blue: 0 },
          },
        },
      ],
      paragraphStyleRange: [
        {
          from: 0,
          to: 100,
          paragraphStyle: {
            burasagari: "burasagariNone",
            singleWordJustification: "justifyAll",
            justificationMethodType: "justifMethodAutomatic",
            textEveryLineComposer: false,
            alignment: "center",
            hangingRoman: true,
            hyphenate: true,
          },
        },
      ],
    },
    typeUnit: "pixelsUnit",
  };
};

const getDefaultStroke = () => {
  return {
    enabled: false,
    size: 0,
    opacity: 100,
    position: "outer",
    color: { r: 255, g: 255, b: 255 },
  };
};

const openFile = (path, autoClose = false) => {
  const encodedPath = JSON.stringify(path);
  csInterface.evalScript(
    "openFile(" + encodedPath + ", " + (autoClose ? "true" : "false") + ")"
  );
};

export { csInterface, locale, openUrl, readStorage, writeToStorage, deleteStorageFile, nativeAlert, nativeConfirm, getUserFonts, getActiveLayerText, setActiveLayerText, getCurrentSelection, getSelectionBoundsHash, startSelectionMonitoring, stopSelectionMonitoring, getSelectionChanged, createTextLayerInSelection, createTextLayersInStoredSelections, alignTextLayerToSelection, alignTextLayerToBubbleAI, fixTrailingSpacesInActiveLayer, aiLoadBubbleModel, aiUnloadBubbleModel, exportActiveDocFlattenPng, createBubbleRectanglesGroup, readBubbleRectanglesGroup, renumberBubbleRectanglesGroup, createTranslatedTextLayers, applyCleanedPngToActiveDoc, openFolderPath, serverCreateJob, serverStartDetectBubbles, serverGetJob, serverGetBubblesAuto, serverSubmitBubbles, serverStartOcrClean, serverGetTranslations, changeActiveLayerTextSize, getHotkeyPressed, resizeTextArea, scrollToLine, scrollToStyle, rgbToHex, getStyleObject, getDefaultStyle, getDefaultStroke, openFile, checkUpdate, downloadAndInstallUpdate };
