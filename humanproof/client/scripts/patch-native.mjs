// Wires the HumanProof native pieces into the generated Capacitor projects.
// Idempotent: safe to run after every `npx cap add` / `npx cap sync`.
//
// Android: Play Integrity plugin, camera/mic permissions, no cleartext, no backup.
// iOS:     App Attest plugin + view controller, usage strings, App Attest entitlement.
import { copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";

const edit = (path, fn) => {
  if (!existsSync(path)) return false;
  const before = readFileSync(path, "utf8");
  const after = fn(before);
  if (after !== before) writeFileSync(path, after);
  return true;
};

function insertAfterLine(text, matcher, lines) {
  const out = [];
  let done = false;
  for (const line of text.split("\n")) {
    out.push(line);
    if (!done && matcher(line)) {
      out.push(...lines);
      done = true;
    }
  }
  if (!done) throw new Error(`anchor not found for: ${lines[0]}`);
  return out.join("\n");
}

// ---------------------------------------------------------------- Android
const ANDROID_PKG = "android/app/src/main/java/com/humanproof/app";
if (existsSync(ANDROID_PKG)) {
  copyFileSync("native/android/HumanProofAttestPlugin.java", `${ANDROID_PKG}/HumanProofAttestPlugin.java`);
  writeFileSync(
    `${ANDROID_PKG}/MainActivity.java`,
    `package com.humanproof.app;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(HumanProofAttestPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
`,
  );
  edit("android/app/src/main/AndroidManifest.xml", (s) => {
    s = s.replace('android:allowBackup="true"', 'android:allowBackup="false"');
    if (!s.includes("usesCleartextTraffic")) {
      s = s.replace('android:allowBackup="false"', 'android:allowBackup="false"\n        android:usesCleartextTraffic="false"');
    }
    if (!s.includes("android.permission.CAMERA")) {
      s = s.replace(
        '<uses-permission android:name="android.permission.INTERNET" />',
        `<uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS" />
    <uses-feature android:name="android.hardware.camera.front" android:required="false" />`,
      );
    }
    return s;
  });
  edit("android/app/build.gradle", (s) =>
    s.includes("play:integrity")
      ? s
      : s.replace(
          "implementation project(':capacitor-android')",
          "implementation project(':capacitor-android')\n    implementation 'com.google.android.play:integrity:1.4.0'",
        ),
  );
  console.log("android: patched");
}

// ---------------------------------------------------------------- iOS
const IOS = "ios/App";
if (existsSync(`${IOS}/App`)) {
  for (const f of ["HumanProofAttestPlugin.swift", "HumanProofViewController.swift"]) {
    copyFileSync(`native/ios/${f}`, `${IOS}/App/${f}`);
  }
  writeFileSync(
    `${IOS}/App/App.entitlements`,
    `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.developer.devicecheck.appattest-environment</key>
  <string>production</string>
</dict>
</plist>
`,
  );

  edit(`${IOS}/App.xcodeproj/project.pbxproj`, (s) => {
    if (s.includes("HumanProofAttestPlugin.swift")) return s;
    const files = [
      ["A1B2C3D4E5F60718293A4B01", "A1B2C3D4E5F60718293A4B02", "HumanProofAttestPlugin.swift"],
      ["A1B2C3D4E5F60718293A4B03", "A1B2C3D4E5F60718293A4B04", "HumanProofViewController.swift"],
    ];
    s = insertAfterLine(
      s,
      (l) => l.includes("SceneDelegate.swift in Sources */ = {isa = PBXBuildFile"),
      files.map(([b, r, n]) => `\t\t${b} /* ${n} in Sources */ = {isa = PBXBuildFile; fileRef = ${r} /* ${n} */; };`),
    );
    s = insertAfterLine(
      s,
      (l) => l.includes("/* SceneDelegate.swift */ = {isa = PBXFileReference"),
      files.map(
        ([, r, n]) =>
          `\t\t${r} /* ${n} */ = {isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = ${n}; sourceTree = "<group>"; };`,
      ),
    );
    s = insertAfterLine(
      s,
      (l) => /^\s+\w+ \/\* SceneDelegate\.swift \*\/,$/.test(l),
      files.map(([, r, n]) => `\t\t\t\t${r} /* ${n} */,`),
    );
    s = insertAfterLine(
      s,
      (l) => /^\s+\w+ \/\* SceneDelegate\.swift in Sources \*\/,$/.test(l),
      files.map(([b, , n]) => `\t\t\t\t${b} /* ${n} in Sources */,`),
    );
    s = s.replaceAll(
      "\t\t\t\tINFOPLIST_FILE = App/Info.plist;",
      "\t\t\t\tCODE_SIGN_ENTITLEMENTS = App/App.entitlements;\n\t\t\t\tINFOPLIST_FILE = App/Info.plist;",
    );
    return s;
  });

  edit(`${IOS}/App/Base.lproj/Main.storyboard`, (s) =>
    s.replace(
      'customClass="CAPBridgeViewController" customModule="Capacitor"',
      'customClass="HumanProofViewController" customModule="App" customModuleProvider="target"',
    ),
  );

  edit(`${IOS}/App/Info.plist`, (s) => {
    if (s.includes("NSCameraUsageDescription")) return s;
    const i = s.lastIndexOf("</dict>");
    return (
      s.slice(0, i) +
      `\t<key>NSCameraUsageDescription</key>
\t<string>HumanProof uses the camera to check that a live person is present. Video never leaves your device.</string>
\t<key>NSMicrophoneUsageDescription</key>
\t<string>HumanProof records a few seconds of you reading five words to check for a live human voice.</string>
` +
      s.slice(i)
    );
  });
  console.log("ios: patched");
}
