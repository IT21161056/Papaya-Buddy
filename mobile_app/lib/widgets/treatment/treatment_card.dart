import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class TreatmentCard extends StatelessWidget {
  final String iconPath;
  final String title;
  final String methodLabel;
  final String methodIconPath;
  final String description;
  final Widget? extraContent;

  const TreatmentCard({
    super.key,
    required this.iconPath,
    required this.title,
    required this.methodLabel,
    required this.methodIconPath,
    required this.description,
    this.extraContent,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: const Color.fromRGBO(115, 236, 139, 1)),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              SvgPicture.asset(iconPath, height: 20, width: 20),
              SizedBox(width: 8),
              Text(
                title,
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          SizedBox(height: 8),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
                color: const Color.fromARGB(103, 203, 255, 144),
                borderRadius: BorderRadius.circular(100),
                border:
                    Border.all(color: const Color.fromARGB(255, 10, 194, 16))),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(methodLabel,
                    style: TextStyle(
                        fontSize: 12,
                        color: const Color.fromARGB(255, 19, 126, 22))),
              ],
            ),
          ),
          SizedBox(height: 8),
          Text(
            description,
            style: TextStyle(fontSize: 14),
          ),
          // Container(
          //   padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          //   decoration:
          //       BoxDecoration(color: const Color.fromARGB(84, 195, 147, 1)),
          //   child:
          // ),
          if (extraContent != null) extraContent!,
        ],
      ),
    );
  }
}
