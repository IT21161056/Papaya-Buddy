import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class TreatmentCard extends StatelessWidget {
  final String iconPath;
  final String title;
  final Color titleColor;
  final String methodLabel;
  final String methodIconPath;
  final Color methodColor;
  final String description;
  final Color cardBackgroundColor;
  final Widget? extraContent;

  const TreatmentCard({
    super.key,
    required this.iconPath,
    required this.title,
    required this.titleColor,
    required this.methodLabel,
    required this.methodIconPath,
    required this.methodColor,
    required this.description,
    required this.cardBackgroundColor,
    this.extraContent,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: cardBackgroundColor,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 8,
            offset: Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: EdgeInsets.all(8), // Adjust padding as needed
                decoration: BoxDecoration(
                  color:
                      titleColor.withOpacity(0.1), // Same color but low opacity
                  borderRadius: BorderRadius.circular(8), // Rounded corners
                ),
                child: SvgPicture.asset(
                  iconPath,
                  height: 24,
                  width: 24,
                  color: titleColor, // Keep icon color the same
                ),
              ),
              SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
            ],
          ),
          SizedBox(height: 16),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: methodColor.withOpacity(0.10),
              borderRadius: BorderRadius.circular(50),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                SvgPicture.asset(methodIconPath,
                    height: 14, width: 16, color: methodColor),
                SizedBox(width: 6),
                Text(
                  methodLabel,
                  style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: methodColor),
                ),
              ],
            ),
          ),
          SizedBox(height: 12),
          Text(
            description,
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w500),
          ),
          if (extraContent != null) ...[
            SizedBox(height: 12),
            extraContent!,
          ],
        ],
      ),
    );
  }
}
