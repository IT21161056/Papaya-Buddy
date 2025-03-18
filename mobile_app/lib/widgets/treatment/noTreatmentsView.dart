import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class NoTreatmentsView extends StatelessWidget {
  final VoidCallback? onRefresh;

  const NoTreatmentsView({Key? key, this.onRefresh}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Empty state illustration
            Container(
              width: 200,
              height: 200,
              decoration: BoxDecoration(
                color: Color.fromRGBO(240, 249, 243, 1),
                borderRadius: BorderRadius.circular(100),
              ),
              child: Center(
                child: Icon(
                  Icons.healing,
                  size: 80,
                  color: Color.fromRGBO(34, 197, 94, 1),
                ),
              ),
            ),

            SizedBox(height: 32),

            // Message title
            Text(
              "No Treatments Available",
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
              "There are currently no treatment options available for this condition.",
              style: TextStyle(
                fontSize: 16,
                color: Colors.black54,
              ),
              textAlign: TextAlign.center,
            ),

            SizedBox(height: 32),

            // Refresh button
            if (onRefresh != null)
              ElevatedButton.icon(
                onPressed: onRefresh,
                icon: Icon(Icons.refresh),
                label: Text("Refresh"),
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
