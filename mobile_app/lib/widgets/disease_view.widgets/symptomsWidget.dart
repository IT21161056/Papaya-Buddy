import 'package:flutter/material.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/models/diseaseModel.dart'; // Import the model

class SymptomsWidget extends StatefulWidget {
  final DiseaseDisplayModel? diseaseModel;
  final int collapsedCount;

  SymptomsWidget({
    Key? key,
    required this.diseaseModel,
    this.collapsedCount = 3,
    required List<String> symptoms,
  }) : super(key: key);

  @override
  State<SymptomsWidget> createState() => _SymptomsWidgetState();
}

class _SymptomsWidgetState extends State<SymptomsWidget> {
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    // Get symptoms from the disease model
    final List<String> symptoms = widget.diseaseModel?.symptoms ?? [];

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.grey[300]!),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.health_and_safety, color: Colors.orange),
              const SizedBox(width: 8),
              const Text(
                "Symptoms",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),

          // Display symptoms - limited by collapsed state
          if (symptoms.isNotEmpty)
            ...List.generate(
              _isExpanded
                  ? symptoms.length
                  : widget.collapsedCount.clamp(0, symptoms.length),
              (index) => _symptomItem(symptoms[index]),
            )
          else
            const Text("No Symptoms Available!"),

          // Show "See more" button if there are more symptoms
          if (symptoms.length > widget.collapsedCount)
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () {
                  setState(() {
                    _isExpanded = !_isExpanded;
                  });
                },
                child: Text(
                  _isExpanded ? "Show less" : "Show more",
                  style: const TextStyle(color: Colors.indigo),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _symptomItem(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(top: 6, right: 8),
            width: 6,
            height: 6,
            decoration: const BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.black87,
            ),
          ),
          Expanded(
            child: Text(
              text,
              style: TextStyle(color: AppColors.textSecondary, fontSize: 14),
            ),
          ),
        ],
      ),
    );
  }
}
