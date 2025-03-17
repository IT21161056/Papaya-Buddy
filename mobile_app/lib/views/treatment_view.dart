import 'package:flutter/material.dart';
import 'package:mobile_app/models/treatmentModel.dart';
import 'package:mobile_app/services/treatmentService.dart';
import 'package:mobile_app/widgets/treatment/treatment_card.dart';

class TreatmentView extends StatefulWidget {
  final String? diseaseId;
  final ValueChanged<bool>? onLoadingChanged; // Optional callback

  TreatmentView({Key? key, this.diseaseId = '', this.onLoadingChanged})
      : super(key: key);

  @override
  _TreatmentViewState createState() => _TreatmentViewState();
}

class _TreatmentViewState extends State<TreatmentView> {
  List<Treatment> treatments = [];
  bool isLoading = true;
  String errorMessage = '';

  @override
  void initState() {
    super.initState();
    _loadPredictions();
  }

  Future<void> _loadPredictions() async {
    setState(() {
      isLoading = true;
      errorMessage = '';
    });

    try {
      if (widget.diseaseId == null || widget.diseaseId!.isEmpty) {
        throw Exception('Disease ID is required');
      }

      List<Treatment>? result =
          await TreatmentService.getTreatmentDataByDisease(widget.diseaseId!);

      if (mounted) {
        setState(() {
          treatments = result ?? [];
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          errorMessage = 'Failed to load treatments: ${e.toString()}';
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8FAFC),
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
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
      ),
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : errorMessage.isNotEmpty
              ? Center(child: Text(errorMessage))
              : treatments.isEmpty
                  ? Center(child: Text('No treatments available'))
                  : SingleChildScrollView(
                      padding: const EdgeInsets.all(20),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Treatment Instructions Button
                          Container(
                            width: double.infinity,
                            padding: EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Color.fromRGBO(220, 252, 231, 1),
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
                          // Loop through treatments to display each treatment
                          ...treatments
                              .map((treatment) => Column(
                                    children: [
                                      TreatmentCard(
                                        iconPath:
                                            'assets/icons/lucide_leaf.svg',
                                        title: treatment.treatmentType,
                                        titleColor:
                                            Color.fromRGBO(34, 197, 94, 1),
                                        methodLabel: treatment.method,
                                        methodIconPath:
                                            'assets/icons/spray.svg',
                                        methodColor:
                                            Color.fromARGB(255, 19, 144, 65),
                                        description: treatment.description,
                                        cardBackgroundColor: Colors.white,
                                        extraContent: Column(
                                          crossAxisAlignment:
                                              CrossAxisAlignment.start,
                                          children: [
                                            Divider(
                                              color: Colors.grey.shade300,
                                              thickness: 1,
                                              height: 20,
                                            ),
                                            _buildInfoSection(
                                              icon: Icons.stars,
                                              title: "Effectiveness",
                                              description:
                                                  treatment.effectiveness,
                                              iconColor: Color.fromRGBO(
                                                  34, 197, 94, 1),
                                            ),
                                            _buildInfoSection(
                                              icon: Icons.warning_amber_rounded,
                                              title: "Side Effects",
                                              description:
                                                  treatment.sideEffects,
                                              iconColor: Color.fromRGBO(
                                                  34, 197, 94, 1),
                                            ),
                                            _buildInfoSection(
                                              icon: Icons.shield,
                                              title: "Precautions",
                                              description:
                                                  treatment.precautions,
                                              iconColor: Color.fromRGBO(
                                                  34, 197, 94, 1),
                                            ),
                                          ],
                                        ),
                                      ),
                                      SizedBox(
                                          height:
                                              16.0), // Adds space between cards
                                    ],
                                  ))
                              .toList(),
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
