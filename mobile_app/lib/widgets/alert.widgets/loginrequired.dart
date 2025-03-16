import 'package:flutter/material.dart';

// Login Alert Widget
class LoginRequiredAlert extends StatelessWidget {
  final VoidCallback? onLoginPressed;
  final VoidCallback? onCancelPressed;

  const LoginRequiredAlert({
    Key? key,
    this.onLoginPressed,
    this.onCancelPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(15),
      ),
      title: Row(
        children: [
          Icon(Icons.account_circle, color: Colors.blue, size: 28),
          SizedBox(width: 8),
          Text('Login Required'),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Image.asset(
            'assets/login_illustration.png', // Replace with your actual asset
            height: 120,
            fit: BoxFit.contain,
            errorBuilder: (context, error, stackTrace) => Icon(
              Icons.login,
              size: 80,
              color: Colors.blue.withOpacity(0.5),
            ),
          ),
          SizedBox(height: 16),
          Text(
            'To use this features you need to Login to our System',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: onCancelPressed ?? () => Navigator.of(context).pop(),
          child: Text('Cancel', style: TextStyle(color: Colors.grey[700])),
        ),
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.blue,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onPressed: onLoginPressed ??
              () {
                Navigator.of(context).pop();
                // Navigate to login screen or show login modal
              },
          child: Text('Login Now'),
        ),
      ],
      actionsPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    );
  }
}

// Example usage functions to show these alerts
void showLoginRequiredAlert(BuildContext context) {
  showDialog(
    context: context,
    builder: (context) => LoginRequiredAlert(
      onLoginPressed: () {
        Navigator.of(context).pop();
        // Navigate to your login screen
        // Example: Navigator.push(context, MaterialPageRoute(builder: (context) => LoginScreen()));
      },
    ),
  );
}
