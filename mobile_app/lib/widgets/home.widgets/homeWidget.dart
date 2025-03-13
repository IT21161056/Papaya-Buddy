// Crop Card Widget
import 'package:flutter/material.dart';

class CropCard extends StatelessWidget {
  final String cropName;
  final IconData icon;
  final VoidCallback onTap;

  const CropCard({
    super.key,
    required this.cropName,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Card(
        elevation: 5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: const LinearGradient(
              colors: [
                Color(0xFFE9F5E9),
                Color(0xFFD4EDDA),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          padding: const EdgeInsets.all(12),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 60,
                    height: 60,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                          color: Colors.green.withOpacity(0.3),
                          blurRadius: 10,
                          spreadRadius: 2,
                        )
                      ],
                    ),
                    child: Center(
                      child: Icon(icon, size: 34, color: Colors.green.shade700),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(cropName,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Color.fromARGB(255, 15, 73, 17),
                      )),
                ],
              ),
              InkWell(
                borderRadius: BorderRadius.circular(20),
                onTap: onTap,
                child: Icon(Icons.arrow_forward_ios,
                    size: 20, color: Color.fromARGB(255, 15, 73, 17)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
