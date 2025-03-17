import 'package:flutter/material.dart';
import 'package:mobile_app/models/diseaseModel.dart'; // Import the model

class DescriptionWidget extends StatefulWidget {
  final DiseaseDisplayModel? diseaseModel;
  final int characterLimit;

  DescriptionWidget({
    Key? key,
    required this.diseaseModel,
    this.characterLimit = 150,
  }) : super(key: key);

  @override
  State<DescriptionWidget> createState() => _DescriptionWidgetState();
}

class _DescriptionWidgetState extends State<DescriptionWidget> {
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    // Get description from the disease model
    final String description =
        widget.diseaseModel?.description ?? 'Description Not Available!';

    // Check if description exceeds limit
    bool descriptionExceedsLimit = description.length > widget.characterLimit;

    // Create the truncated text version
    String truncatedText = descriptionExceedsLimit
        ? description.substring(0, widget.characterLimit) + "..."
        : description;

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
          const Text(
            "Description",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          Text(
            _isExpanded ? description : truncatedText,
            style: const TextStyle(fontSize: 14),
          ),
          // Only show "See more" button if text exceeds limit
          if (descriptionExceedsLimit)
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () {
                  setState(() {
                    _isExpanded = !_isExpanded;
                  });
                },
                child: Text(
                  _isExpanded ? "Show less" : "See more...",
                  style: const TextStyle(color: Colors.indigo),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
