import 'package:flutter/material.dart';

class NoPredictionsView extends StatelessWidget {
  final VoidCallback? onScanPressed;

  const NoPredictionsView({Key? key, this.onScanPressed}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Container(
              width: 200,
              height: 200,
              decoration: BoxDecoration(
                color: Color.fromRGBO(240, 249, 243, 1),
                borderRadius: BorderRadius.circular(100),
              ),
              child: Center(
                child: Icon(
                  Icons.eco,
                  size: 80,
                  color: Color.fromRGBO(34, 197, 94, 1),
                ),
              ),
            ),

            SizedBox(height: 32),

            // Message title
            Text(
              "No Plant Diagnoses Yet",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
              textAlign: TextAlign.center,
            ),

            SizedBox(height: 16),

            // Message description
            Text(
              "You haven't scanned any plants yet. Start by scanning your first plant to get a diagnosis!",
              style: TextStyle(
                fontSize: 16,
                color: Colors.black54,
              ),
              textAlign: TextAlign.center,
            ),

            SizedBox(height: 32),

            ElevatedButton.icon(
              onPressed: onScanPressed,
              icon: Icon(Icons.camera_alt),
              label: Text("Scan Plant"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Color.fromRGBO(34, 197, 94, 1),
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
