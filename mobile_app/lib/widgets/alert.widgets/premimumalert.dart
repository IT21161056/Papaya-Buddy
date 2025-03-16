import 'package:flutter/material.dart';

// Premium Subscription Alert Widget
class PremiumRequiredAlert extends StatelessWidget {
  final VoidCallback? onSubscribePressed;
  final VoidCallback? onCancelPressed;

  const PremiumRequiredAlert({
    Key? key,
    this.onSubscribePressed,
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
          Icon(Icons.star, color: Colors.amber, size: 28),
          SizedBox(width: 8),
          Text('Premium Feature'),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.amber.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              Icons.workspace_premium,
              size: 80,
              color: Colors.amber,
            ),
          ),
          SizedBox(height: 16),
          Text(
            'To use this feature, you need to subscribe to our premium plan',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16),
          ),
          SizedBox(height: 8),
          Text(
            'Get unlimited access to all premium features',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: onCancelPressed ?? () => Navigator.of(context).pop(),
          child: Text('Not now', style: TextStyle(color: Colors.grey[700])),
        ),
        ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.amber[700],
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onPressed: onSubscribePressed ??
              () {
                Navigator.of(context).pop();
                // Navigate to subscription screen
              },
          child: Text('Subscribe Now'),
        ),
      ],
      actionsPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    );
  }
}

void showPremiumRequiredAlert(BuildContext context) {
  showDialog(
    context: context,
    builder: (context) => PremiumRequiredAlert(
      onSubscribePressed: () {
        Navigator.of(context).pop();
        // Navigate to your subscription screen
        // Example: Navigator.push(context, MaterialPageRoute(builder: (context) => SubscriptionScreen()));
      },
    ),
  );
}
