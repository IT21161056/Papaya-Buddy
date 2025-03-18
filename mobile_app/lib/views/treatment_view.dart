import 'package:flutter/material.dart';
import 'package:mobile_app/models/treatmentModel.dart';
import 'package:mobile_app/services/treatmentService.dart';
import 'package:mobile_app/widgets/treatment/noTreatmentsView.dart';
import 'package:mobile_app/widgets/treatment/treatment_card.dart';
import 'package:mobile_app/utils/helper.dart';
import 'package:flutter_svg/flutter_svg.dart';

class TreatmentView extends StatefulWidget {
  final String? diseaseId;
  final ValueChanged<bool>? onLoadingChanged;

  TreatmentView({Key? key, this.diseaseId = '', this.onLoadingChanged})
      : super(key: key);

  @override
  _TreatmentViewState createState() => _TreatmentViewState();
}

class _TreatmentViewState extends State<TreatmentView> {
  List<Treatment> treatments = [];
  bool isLoading = true;
  String errorMessage = '';

  Map<String, dynamic> _getTreatmentTypeStyles(String treatmentType) {
    switch (treatmentType.toLowerCase()) {
      case "chemical":
        return {
          'color': const Color.fromARGB(255, 59, 100, 246),
          'iconPath': 'assets/icons/flusk.svg',
        };
      case "biological":
        return {
          'color': const Color.fromRGBO(34, 197, 94, 1),
          'iconPath': 'assets/icons/lucide_leaf.svg',
        };
      case "cultural":
        return {
          'color': const Color.fromRGBO(168, 85, 247, 1),
          'iconPath': 'assets/icons/user.svg',
        };
      case "mechanical":
        return {
          'color': const Color.fromRGBO(245, 158, 11, 1),
          'iconPath': 'assets/icons/octicon_tools.svg',
        };
      case "organic":
        return {
          'color': const Color.fromRGBO(16, 185, 129, 1),
          'iconPath': 'assets/icons/lucide_leaf.svg',
        };
      default:
        return {
          'color': const Color.fromRGBO(75, 85, 99, 1),
          'iconPath': 'assets/icons/lucide_circle_help.svg',
        };
    }
  }

  @override
  void initState() {
    super.initState();
    _loadTreatments();
  }

  Future<void> _loadTreatments() async {
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
                  ? NoTreatmentsView(onRefresh: _loadTreatments)
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
                              color: Colors.blue[50],
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Row(
                              children: [
                                //
                                SvgPicture.asset(
                                    'icon_park_outline_instruction.svg',
                                    height: 14,
                                    width: 16,
                                    color: Colors.blue),
                                Text(
                                  "Treatment Instructions",
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w600,
                                    color: Colors.blue,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          SizedBox(height: 16),

                          ...treatments.map((treatment) {
                            final styles = _getTreatmentTypeStyles(
                                treatment.treatmentType);

                            return Column(
                              children: [
                                TreatmentCard(
                                  iconPath: styles['iconPath'],
                                  title:
                                      capitalizeText(treatment.treatmentType),
                                  titleColor: styles['color'],
                                  methodLabel: treatment.method,
                                  methodIconPath: 'assets/icons/spray.svg',
                                  methodColor: styles['color'].withOpacity(0.8),
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
                                        description: treatment.effectiveness,
                                        iconColor: styles['color'],
                                      ),
                                      _buildInfoSection(
                                        icon: Icons.warning_amber_rounded,
                                        title: "Side Effects",
                                        description: treatment.sideEffects,
                                        iconColor: styles['color'],
                                      ),
                                      _buildInfoSection(
                                        icon: Icons.shield,
                                        title: "Precautions",
                                        description: treatment.precautions,
                                        iconColor: styles['color'],
                                      ),
                                    ],
                                  ),
                                ),
                                SizedBox(
                                    height: 16.0), // Adds space between cards
                              ],
                            );
                          }).toList(),
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
