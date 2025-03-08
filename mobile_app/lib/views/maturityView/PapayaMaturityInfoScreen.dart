import 'package:flutter/material.dart';

class PapayaMaturityInfoScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Papaya Maturity Levels")),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch, // Ensures equal width
          children: [
            _buildMaturityCard(
              "Not Mature",
              "Completely green, needs 7-10 days to ripen.",
              Colors.teal,
            ),
            _buildMaturityCard(
              "Partially Mature",
              "Green with yellow spots, 3-5 days to ripen.",
              Colors.lightGreen,
            ),
            _buildMaturityCard(
              "Mature",
              "Mostly yellow with some green, ready to eat.",
              Colors.green,
            ),
            _buildMaturityCard(
              "Rotten",
              "Fully yellow with soft spots, overripe or spoiled.",
              Colors.yellowAccent,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMaturityCard(String level, String description, Color color) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 4,
      color: color.withOpacity(0.7),
      child: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment:
              CrossAxisAlignment.start, // Align text to the left
          children: [
            Text(
              level,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.black,
              ),
            ),
            SizedBox(height: 8),
            Text(
              description,
              style: TextStyle(fontSize: 16, color: Colors.black87),
            ),
          ],
        ),
      ),
    );
  }
}
