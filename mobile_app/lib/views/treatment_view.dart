import 'package:flutter/material.dart';
import 'package:mobile_app/models/treatmentModel.dart';
import 'package:mobile_app/services/treatmentService.dart';
import 'package:mobile_app/widgets/treatment/treatment_card.dart';

class TreatmentView extends StatefulWidget {
  final String? treatmentId;
  final String? diseaseId;

  TreatmentView({Key? key, this.treatmentId, this.diseaseId}) : super(key: key);

  @override
  _TreatmentViewState createState() => _TreatmentViewState();
}

class _TreatmentViewState extends State<TreatmentView> {
  late Future<List<Treatment>?> treatments;

  @override
  void initState() {
    super.initState();
    // Initialize the future with the fetch call
    treatments =
        TreatmentService.getTreatmentDataByDisease(widget.diseaseId ?? '');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8FAFC), // Light background color
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
          alignment: Alignment.center,
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
      body: FutureBuilder<List<Treatment>?>(
        future: treatments,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: CircularProgressIndicator());
          } else if (snapshot.hasError) {
            return Center(child: Text('Error fetching data'));
          } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
            return Center(child: Text('No treatments available'));
          } else {
            List<Treatment> treatments = snapshot.data!;

            return SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Treatment Instructions Button
                  Container(
                    width: double.infinity,
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Color.fromRGBO(
                          220, 252, 231, 1), // Light green background
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
                      .map((treatment) => TreatmentCard(
                            iconPath: 'assets/icons/lucide_leaf.svg',
                            title: treatment.treatmentType.join(', '),
                            titleColor: Color.fromRGBO(34, 197, 94, 1),
                            methodLabel: treatment.method,
                            methodIconPath: 'assets/icons/spray.svg',
                            methodColor: Color.fromARGB(255, 19, 144, 65),
                            description: treatment.description,
                            cardBackgroundColor: Colors.white,
                            extraContent: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
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
                                      treatment.effectiveness.join(', '),
                                  iconColor: Color.fromRGBO(34, 197, 94, 1),
                                ),
                                _buildInfoSection(
                                  icon: Icons.warning_amber_rounded,
                                  title: "Side Effects",
                                  description: treatment.sideEffects,
                                  iconColor: Color.fromRGBO(34, 197, 94, 1),
                                ),
                                _buildInfoSection(
                                  icon: Icons.shield,
                                  title: "Precautions",
                                  description: treatment.precautions,
                                  iconColor: Color.fromRGBO(34, 197, 94, 1),
                                ),
                              ],
                            ),
                          ))
                      .toList(),
                ],
              ),
            );
          }
        },
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
