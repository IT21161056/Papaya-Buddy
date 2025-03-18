import 'package:flutter/material.dart';
import 'package:mobile_app/theme/colors.dart';

class UserNotLoggedInView extends StatelessWidget {
  final VoidCallback? onLoginPressed;

  const UserNotLoggedInView({Key? key, this.onLoginPressed}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text(
              "Sign In Required",
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 16),
            Text(
              "Please sign in to view your plant diagnosis history.",
              style: TextStyle(
                fontSize: 14,
                color: Colors.black54,
              ),
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 32),
            ElevatedButton.icon(
              onPressed: onLoginPressed,
              icon: Icon(
                Icons.login,
                color: Colors.white,
              ),
              label: Text(
                "Sign In",
                style: TextStyle(),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
