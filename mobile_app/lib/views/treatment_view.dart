import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/widgets/treatment/treatment_card.dart';

class TreatmentScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: Row(
          children: [
            Text(
              "Treatments",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: Colors.black,
                fontSize: 20,
              ),
            ),
            // Spacer(),
            SizedBox(width: 8),
            SvgPicture.asset(
              'assets/icons/spray_can.svg',
              height: 24,
              width: 24,
            ),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Treatment Instructions Button
            Container(
              padding: EdgeInsets.symmetric(vertical: 10, horizontal: 16),
              decoration: BoxDecoration(
                color: Colors.lightGreenAccent.shade100,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                "Treatment Instructions",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
              ),
            ),
            SizedBox(height: 16),

            // Organic Control Card
            TreatmentCard(
              iconPath: 'assets/icons/leaf.svg', // Replace with actual asset
              title: "Organic Control",
              methodLabel: "Spray method",
              methodIconPath: 'assets/icons/spray.svg',
              description:
                  "Spray a mix of baking soda, water, and neem oil weekly to control powdery mildew organically.",
            ),

            SizedBox(height: 16),

            // Chemical Control Card
            TreatmentCard(
              iconPath:
                  'assets/icons/chemical.svg', // Replace with actual asset
              title: "Chemical Control",
              methodLabel: "Spray method",
              methodIconPath: 'assets/icons/spray.svg',
              description: "Thiophanate-Methyl 70.0% WP",
              extraContent: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: 8),
                  Row(
                    children: [
                      Icon(Icons.air, size: 20, color: Colors.black),
                      SizedBox(width: 8),
                      Text(
                        "Weather Conditions",
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 4),
                  Text(
                    "Avoid applying during wind, rain, or extreme heat for best results.",
                    style: TextStyle(fontSize: 14),
                  ),
                  SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: () {},
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.lightGreenAccent.shade100,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: Text(
                      "See how to use",
                      style: TextStyle(color: Colors.black),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
