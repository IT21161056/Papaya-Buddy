import 'package:flutter/material.dart';
import 'package:mobile_app/models/predictionModel.dart';
import 'package:mobile_app/theme/colors.dart';

class PredictionDetailScreen extends StatelessWidget {
  final Prediction prediction;

  PredictionDetailScreen({required this.prediction});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // Header with Back Button and Title
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(
                  bottom: BorderSide(
                    color: Color(0xFFF1F5F9),
                    width: 1,
                  ),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Back Button
                  Material(
                    color: const Color(0xFFF8FAFC),
                    borderRadius: BorderRadius.circular(12),
                    child: InkWell(
                      onTap: () => Navigator.pop(context),
                      borderRadius: BorderRadius.circular(12),
                      child: Container(
                        width: 40,
                        height: 40,
                        alignment: Alignment.center,
                        child: const Icon(
                          Icons.chevron_left,
                          size: 24,
                          color: Color(0xFF1A1A1A),
                        ),
                      ),
                    ),
                  ),
                  // Title
                  const Text(
                    'Prediction Details',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A1A1A),
                    ),
                  ),
                  // Placeholder for spacing
                  const SizedBox(width: 40),
                ],
              ),
            ),

            Expanded(
              child: Container(
                color: const Color(0xFFF8FAFC), // Background color
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12.0),
                        child: Image.network(
                          prediction.uploadedImgUrl,
                          width: double.infinity,
                          height: 200,
                          fit: BoxFit.cover,
                        ),
                      ),
                      const SizedBox(height: 20),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.05),
                              blurRadius: 5,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              prediction.disease.name,
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: AppColors.textPrimary,
                              ),
                            ),
                            const SizedBox(height: 10),
                            Text(
                              prediction.disease.description,
                              style: TextStyle(
                                fontSize: 16,
                                color: AppColors.textSecondary,
                              ),
                              textAlign: TextAlign.justify,
                            ),
                            const SizedBox(height: 20),
                            Row(
                              children: [
                                Icon(
                                  Icons.calendar_today,
                                  size: 18,
                                  color: Colors.grey.shade500,
                                ),
                                const SizedBox(width: 8), // Spacing
                                Text(
                                  'Predicted on: ${prediction.createdAt.toLocal()}',
                                  style: TextStyle(
                                    fontSize: 14,
                                    color: Colors.grey.shade500,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      )
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
