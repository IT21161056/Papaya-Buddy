import 'package:flutter/material.dart';

class RecommendationsSection extends StatefulWidget {
  @override
  _RecommendationsSectionState createState() => _RecommendationsSectionState();
}

class _RecommendationsSectionState extends State<RecommendationsSection> {
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(10),
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 5,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                "Recommendations",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          SizedBox(height: 10),
          Text(
            "Tips to keep your plant healthy",
            style: TextStyle(fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 5),
          Text(
              "• Fertilize with the right fertilizer mixture and a balanced nutrient supply"),
          Text("• Do not over-water the crop during the season"),
          Text("• Do not touch healthy plants after touching infected plants"),
          Text(
              "• Maintain a high number of different varieties of plants around fields"),
          SizedBox(height: 5),
          if (_isExpanded) ...[
            Text(
                "• Provide proper spacing between plants to allow air circulation"),
            Text("• Regularly check for pests and diseases"),
            Text("• Use organic or chemical pest control methods as needed"),
          ],
          SizedBox(height: 5),
          GestureDetector(
            onTap: () {
              setState(() {
                _isExpanded = !_isExpanded;
              });
            },
            child: Text(
              _isExpanded ? "Show less" : "Show more",
              style: TextStyle(color: Colors.blue),
            ),
          ),
        ],
      ),
    );
  }
}
