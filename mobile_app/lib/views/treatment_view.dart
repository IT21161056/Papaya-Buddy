import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/widgets/treatment/treatment_card.dart';

class TreatmentScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8FAFC), // Light background color
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
          alignment: Alignment.center,
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        title: Text(
          "Treatments",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.black,
            fontSize: 20,
          ),
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 16),
            child: SvgPicture.asset(
              'assets/icons/leaf.svg',
              height: 24,
              width: 24,
            ),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Treatment Instructions Button
            Container(
              width: double.infinity,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color:
                    Color.fromRGBO(220, 252, 231, 1), // Light green background
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                "Treatment Instructions",
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color.fromRGBO(22, 101, 52, 1),
                ),
              ),
            ),
            SizedBox(height: 16),

            // Organic Control Card
            TreatmentCard(
              iconPath: 'assets/icons/lucide_leaf.svg',
              title: "Organic Control",
              titleColor: Color.fromRGBO(34, 197, 94, 1), // Green title
              methodLabel: "Spray method",
              methodIconPath: 'assets/icons/spray.svg',
              methodColor: Color.fromARGB(255, 19, 144, 65), // Green badge
              description:
                  "Spray a mix of baking soda, water, and neem oil weekly to control powdery mildew organically.",
              cardBackgroundColor: Colors.white, // White card
              extraContent: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Divider(
                    color: Colors.grey.shade300,
                    thickness: 1,
                    height: 20,
                  ),
                  _buildInfoSection(
                    icon: Icons.stars,
                    title: "Effectiveness",
                    description: "Low",
                    iconColor: Color.fromRGBO(34, 197, 94, 1),
                  ),
                  _buildInfoSection(
                    icon: Icons.warning_amber_rounded,
                    title: "Side Effects",
                    description: "Strong odor, may need frequent application.",
                    iconColor: Color.fromRGBO(34, 197, 94, 1),
                  ),
                  _buildInfoSection(
                    icon: Icons.shield,
                    title: "Precautions",
                    description:
                        "Apply during early morning or late evening to avoid leaf burn.",
                    iconColor: Color.fromRGBO(34, 197, 94, 1),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // Chemical Control Card
            TreatmentCard(
              iconPath: 'assets/icons/flusk.svg',
              title: "Chemical Control",
              titleColor: Color(0xFF6366F1), // Purple title
              methodLabel: "Spray method",
              methodIconPath: 'assets/icons/spray.svg',
              methodColor: Color(0xFF6366F1), // Purple badge
              description: "Thiophanate-Methyl 70.0% WP",
              cardBackgroundColor: Colors.white, // White card
              extraContent: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Divider(
                    color: Colors.grey.shade300,
                    thickness: 1,
                    height: 20,
                  ),
                  _buildInfoSection(
                    icon: Icons.stars,
                    title: "Effectiveness",
                    description: "High",
                    iconColor: Color(0xFF6366F1),
                  ),
                  _buildInfoSection(
                    icon: Icons.warning_amber_rounded,
                    title: "Side Effects",
                    description:
                        "Potential toxicity to beneficial insects, environmental impact.",
                    iconColor: Color(0xFF6366F1),
                  ),
                  _buildInfoSection(
                    icon: Icons.shield,
                    title: "Precautions",
                    description:
                        "Use protective gear, avoid application near water sources.",
                    iconColor: Color(0xFF6366F1),
                  ),
                  SizedBox(height: 12),
                  Row(
                    children: [
                      Icon(Icons.cloud, size: 20, color: Colors.black),
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
                  SizedBox(height: 12),
                  Text(
                    "Avoid applying during wind, rain, or extreme heat for best results.",
                    style: TextStyle(fontSize: 14),
                  ),
                  SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity, // Makes button full width
                    child: ElevatedButton(
                      onPressed: () {
                        // Add your action here
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        padding: EdgeInsets.symmetric(vertical: 20),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 0,
                      ),
                      child: Text(
                        "Apply Treatment",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
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

  // Helper method to build info sections
  Widget _buildInfoSection({
    required IconData icon,
    required String title,
    required String description,
    required Color iconColor,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 20, color: iconColor),
              SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          SizedBox(height: 8),
          Text(
            description,
            style: TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
}
