import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AuthService {
  static final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // Add this constant for subscription plan IDs
  static const String freePlanId = 'fPayFpAmqocLTmtzC1r3';

  Future<User?> signUp({
    required String email,
    required String password,
    required String fullName,
    required String phoneNumber,
    required String city,
  }) async {
    try {
      UserCredential userCredential =
          await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );

      User? user = userCredential.user;

      if (user != null) {
        // Updated user document with subscription info
        await _firestore.collection("users").doc(user.uid).set({
          "fullName": fullName,
          "email": email,
          "phoneNumber": phoneNumber,
          "city": city,
          "uid": user.uid,
          "active_plan_id": freePlanId, // Assign free plan by default
          "subscription_expiry": null, // No expiry for free plan
          "created_at": FieldValue.serverTimestamp(),
        });
      }

      return user;
    } catch (e) {
      print("Signup Error: $e");
      return null;
    }
  }

  Future<Map<String, dynamic>?> getSubscriptionDetails() async {
    User? user = _auth.currentUser;
    if (user != null) {
      DocumentSnapshot doc =
          await _firestore.collection("users").doc(user.uid).get();
      if (doc.exists) {
        final userData = doc.data() as Map<String, dynamic>;
        return {
          'active_plan_id': userData['active_plan_id'] ?? freePlanId,
          'subscription_expiry': userData['subscription_expiry'],
        };
      }
    }
    return null;
  }

  Future<User?> signIn(
      {required String email, required String password}) async {
    try {
      UserCredential userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      return userCredential.user;
    } catch (e) {
      print("Sign In Error: $e");
      return null;
    }
  }

  // Sign Out
  Future<void> signOut() async {
    await _auth.signOut();
  }

  User? getCurrentUser() {
    return _auth.currentUser;
  }

  String? getUserUID() {
    return _auth.currentUser?.uid;
  }

  // Get user details from Firestore
  Future<Map<String, dynamic>?> getUserDetails() async {
    User? user = _auth.currentUser;
    if (user != null) {
      DocumentSnapshot doc =
          await _firestore.collection("users").doc(user.uid).get();
      if (doc.exists) {
        return doc.data() as Map<String, dynamic>;
      }
    }
    return null;
  }

  // Updated Google Sign-In to include subscription info
  Future<User?> signInWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await GoogleSignIn().signIn();
      if (googleUser == null) return null;

      final GoogleSignInAuthentication googleAuth =
          await googleUser.authentication;

      final OAuthCredential credential = GoogleAuthProvider.credential(
        accessToken: googleAuth.accessToken,
        idToken: googleAuth.idToken,
      );

      UserCredential userCredential =
          await _auth.signInWithCredential(credential);

      if (userCredential.user != null) {
        // Check if user exists, if not create with free plan
        final userDoc = await _firestore
            .collection("users")
            .doc(userCredential.user!.uid)
            .get();
        if (!userDoc.exists) {
          await _firestore
              .collection("users")
              .doc(userCredential.user!.uid)
              .set({
            "fullName": googleUser.displayName,
            "email": googleUser.email,
            "phoneNumber": "",
            "city": "",
            "uid": userCredential.user!.uid,
            "active_plan_id": freePlanId,
            "subscription_expiry": null,
            "created_at": FieldValue.serverTimestamp(),
          });
        }
      }

      return userCredential.user;
    } catch (e) {
      print("Google Sign-In Error: $e");
      return null;
    }
  }
}
