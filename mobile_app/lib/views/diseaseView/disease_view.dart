import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:mobile_app/models/diseaseModel.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/views/treatment_view.dart';
import 'package:mobile_app/widgets/disease_view.widgets/descriptionWidget.dart';
import 'package:mobile_app/widgets/disease_view.widgets/symptomsWidget.dart';

class DiseaseView extends StatefulWidget {
  final String? diseaseId;
  final DiseaseDisplayModel? disease;

  DiseaseView({Key? key, this.diseaseId, this.disease}) : super(key: key);

  @override
  _DiseaseViewState createState() => _DiseaseViewState();
}

class _DiseaseViewState extends State<DiseaseView> {
  int _currentImageIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Diagnosis"),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
          alignment: Alignment.center,
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () {},
          ),
          IconButton(
            icon: const Icon(Icons.more_vert),
            onPressed: () {},
          ),
        ],
      ),
      backgroundColor: AppColors.background,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title and disease type
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.disease?.name ?? 'Disease Name',
                        style: const TextStyle(
                            fontSize: 22, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        widget.disease?.diseaseType ?? 'Disease Type',
                        style:
                            const TextStyle(fontSize: 16, color: Colors.green),
                      ),
                    ],
                  ),
                ),
                const Icon(
                  Icons.bug_report,
                  color: Colors.green,
                  size: 24,
                ),
              ],
            ),

            const SizedBox(height: 12),

            // Image carousel
            Stack(
              children: [
                CarouselSlider(
                  options: CarouselOptions(
                    height: 200,
                    enableInfiniteScroll:
                        (widget.disease?.suggestedImageUrls.length ?? 0) > 1,
                    enlargeCenterPage: true,
                    autoPlay:
                        (widget.disease?.suggestedImageUrls.length ?? 0) > 1,
                    onPageChanged: (index, reason) {
                      setState(() {
                        _currentImageIndex = index;
                      });
                    },
                  ),
                  items: widget.disease?.suggestedImageUrls.isNotEmpty == true
                      ? widget.disease!.suggestedImageUrls.map((path) {
                          return Container(
                            width: MediaQuery.of(context).size.width,
                            margin: const EdgeInsets.symmetric(horizontal: 2.0),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: Image.network(
                                path,
                                fit: BoxFit.cover,
                                loadingBuilder:
                                    (context, child, loadingProgress) {
                                  if (loadingProgress == null) return child;
                                  return Center(
                                    child: CircularProgressIndicator(
                                      value:
                                          loadingProgress.expectedTotalBytes !=
                                                  null
                                              ? loadingProgress
                                                      .cumulativeBytesLoaded /
                                                  loadingProgress
                                                      .expectedTotalBytes!
                                              : null,
                                    ),
                                  );
                                },
                                errorBuilder: (context, error, stackTrace) {
                                  return Container(
                                      color: Colors.grey[300],
                                      child: const Center(
                                        child: Column(
                                          mainAxisAlignment:
                                              MainAxisAlignment.center,
                                          children: [
                                            Icon(Icons.error,
                                                color: Colors.red),
                                            Text("Failed to load Image!"),
                                          ],
                                        ),
                                      ));
                                },
                              ),
                            ),
                          );
                        }).toList()
                      : [
                          Container(
                            width: MediaQuery.of(context).size.width,
                            color: Colors.grey[300],
                            child: const Center(
                                child: Text("No images available...!")),
                          )
                        ],
                ),
                if (widget.disease?.suggestedImageUrls.isNotEmpty == true)
                  Positioned(
                    right: 10,
                    bottom: 10,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.8),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        "${_currentImageIndex + 1}/${widget.disease?.suggestedImageUrls.length} photos",
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  ),
              ],
            ),

            const SizedBox(height: 16),

            // Information box
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  const Icon(Icons.info_outline, color: Colors.blue),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.disease?.preventiveMeasures ??
                          'Preventive measures not available!',
                      style: TextStyle(color: AppColors.textSecondary),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // Description section with character limit
            DescriptionWidget(
              diseaseModel: widget.disease,
              characterLimit: 150,
            ),

            const SizedBox(height: 16),

            // Symptoms section
            SymptomsWidget(
              symptoms: widget.disease?.symptoms ?? [],
              collapsedCount: 3,
              diseaseModel: widget.disease, // Using the same value as before
            ),

            const SizedBox(height: 16),

            // Treatment button
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => TreatmentScreen(
                        // diseaseId: widget.disease?.id,
                        // diseaseName: widget.disease?.name,
                        ),
                  ),
                );
              },
              child: const Text(
                "Treatment Instructions",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),

            const SizedBox(height: 12),

            // Save button
            OutlinedButton(
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black87,
                minimumSize: const Size(double.infinity, 50),
                side: BorderSide(color: Colors.grey[300]!),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () {},
              child: const Text(
                "Save to Diagnoses",
                style: TextStyle(fontSize: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
