# Setting Up Firebase & FlutterFire CLI for Your Flutter Project

This guide will walk you through the process of installing Firebase tools, configuring FlutterFire CLI, and integrating Firebase into your Flutter project.

---

## Prerequisites

Ensure you have the following installed before proceeding:

- **Node.js** (for Firebase CLI) → [Download Here](https://nodejs.org/)
- **Flutter SDK** → [Installation Guide](https://flutter.dev/docs/get-started/install)
- **Dart** (Comes with Flutter)

---

## 1️⃣ Install Firebase CLI

The Firebase CLI allows you to manage Firebase projects from the command line.

```sh
npm i -g firebase-tools
```

## 2️⃣ Login to Firebase

Authenticate with your Google account to access Firebase services.

```sh
firebase login
```

Follow the on-screen instructions to log in.

## 3️⃣ Install FlutterFire CLI

FlutterFire CLI is required to configure Firebase in your Flutter project.

```sh
dart pub global activate flutterfire_cli
```

### Add FlutterFire CLI to Your System Path

Since `flutterfire_cli` is installed globally, you need to add it to your system's PATH.

#### **For Zsh Users:**

```sh
echo 'export PATH="$PATH:$HOME/.pub-cache/bin"' >> ~/.zshrc
source ~/.zshrc
```

#### **For Bash Users:**

```sh
echo 'export PATH="$PATH:$HOME/.pub-cache/bin"' >> ~/.bash_profile
source ~/.bash_profile
```

## 4️⃣ Verify Installation

Check if `flutterfire_cli` is installed correctly by running:

```sh
flutterfire --version
```

If the installation is successful, it will return the installed version number.

---

## 5️⃣ Add Firebase to Your Flutter Project

### **Step 1: Add Firebase Core Dependency**

Firebase requires the `firebase_core` package to initialize services.

```sh
flutter pub add firebase_core
```

### **Step 2: Configure Firebase**

Run the following command to link your Flutter project with Firebase:

```sh
flutterfire configure
```

- You will see a list of existing Firebase projects or the option to create a new one.
- Select the appropriate Firebase project.
- FlutterFire will automatically generate the required configuration files (`google-services.json` for Android, `GoogleService-Info.plist` for iOS/macOS).

---

## ✅ Next Steps

After configuring Firebase, you can proceed with integrating Firebase services like Authentication, Firestore, or Storage into your Flutter app.

For further details, refer to the [FlutterFire Documentation](https://firebase.flutter.dev/). 🚀

Platform Firebase App Id
web 1:605393192584:web:16b672c8265fa6cc6bc13d
android 1:605393192584:android:9a6ebda5e288db8d6bc13d
ios 1:605393192584:ios:720d8997ed452d626bc13d
macos 1:605393192584:ios:720d8997ed452d626bc13d
windows 1:605393192584:web:62d4bbe438dd66c36bc13d
