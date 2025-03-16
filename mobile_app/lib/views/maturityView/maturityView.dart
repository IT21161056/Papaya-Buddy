import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:mobile_app/models/diseaseModel.dart';
import 'package:mobile_app/models/maturityModel.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/views/maturityView/PapayaMaturityInfoScreen.dart';
import 'package:mobile_app/views/treatment_view.dart';

class MaturityView extends StatefulWidget {
  final MaturityStageDisplayModel? maturityStage;

  MaturityView({Key? key, this.maturityStage}) : super(key: key);

  @override
  _MaturityViewState createState() => _MaturityViewState();
}

class _MaturityViewState extends State<MaturityView> {
  bool _isSymptomExpanded = false;
  bool _isCauseExpanded = false;
  int _currentImageIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text(
          "Maturity Stage",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.black,
            fontSize: 20,
          ),
        ),
        centerTitle: true,
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
          alignment: Alignment.center,
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      backgroundColor: AppColors.background,
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Title and maturity stage
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        widget.maturityStage?.stage ?? 'Mature',
                        style: TextStyle(
                            fontSize: 22, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.eco,
                  color: Colors.green,
                  size: 24,
                ),
              ],
            ),

            SizedBox(height: 12),

            // Image carousel
            Stack(
              children: [
                CarouselSlider(
                  options: CarouselOptions(
                    height: 200,
                    enableInfiniteScroll: true,
                    enlargeCenterPage: true,
                    autoPlay: true,
                    onPageChanged: (index, reason) {
                      setState(() {
                        _currentImageIndex = index;
                      });
                    },
                  ),
                  items: widget.maturityStage?.suggestedImageUrls.map((path) {
                        return Container(
                          width: MediaQuery.of(context).size.width,
                          margin: EdgeInsets.symmetric(horizontal: 5.0),
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
                                    value: loadingProgress.expectedTotalBytes !=
                                            null
                                        ? loadingProgress
                                                .cumulativeBytesLoaded /
                                            loadingProgress.expectedTotalBytes!
                                        : null,
                                  ),
                                );
                              },
                              errorBuilder: (context, error, stackTrace) {
                                return Container(
                                  color: Colors.grey[300],
                                  child: const Center(
                                    child: Icon(Icons.error, color: Colors.red),
                                  ),
                                );
                              },
                            ),
                          ),
                        );
                      }).toList() ??
                      [
                        Container(
                          width: MediaQuery.of(context).size.width,
                          color: Colors.grey[300],
                          child: Center(child: Text("No images available")),
                        )
                      ],
                ),
                Positioned(
                  right: 10,
                  bottom: 10,
                  child: Container(
                    padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.8),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      "${_currentImageIndex + 1}/${widget.maturityStage?.suggestedImageUrls.length ?? 0} photos",
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                ),
              ],
            ),

            SizedBox(height: 16),

            // Best time to harvest notice
            Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.green[50],
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Icon(Icons.timer, color: Colors.green),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.maturityStage?.bestTimeToHarvest ??
                          'Ideal time to harvest for best flavor and texture',
                      style: TextStyle(color: Colors.green[900]),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // Description section
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Description",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 12),
                  Text(
                    widget.maturityStage?.description ??
                        'The fruit has reached its typical color. Skin yields slightly to pressure and has a sweet aroma. The flesh is juicy and flavorful.',
                    style: TextStyle(fontSize: 14),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // Timing information with view other stages button
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        "Timing",
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      ElevatedButton.icon(
                        icon: Icon(Icons.view_timeline, size: 16),
                        label: Text("View All Stages"),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green[100],
                          foregroundColor: Colors.green[800],
                          padding:
                              EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                          textStyle: TextStyle(fontSize: 12),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                        onPressed: () {
                          // Navigate to all maturity stages view
                          // This will be implemented in the future
                          showDialog(
                            context: context,
                            builder: (BuildContext context) {
                              return AlertDialog(
                                title: Text("Maturity Stages"),
                                content: Text(
                                    "This will show all maturity stages in a timeline view."),
                                actions: [
                                  TextButton(
                                    onPressed: () {
                                      Navigator.pop(context);
                                    },
                                    child: Text("OK"),
                                  ),
                                ],
                              );
                            },
                          );
                        },
                      ),
                    ],
                  ),
                  SizedBox(height: 12),
                  _infoRow(
                    Icons.calendar_today,
                    "Time to Reach",
                    widget.maturityStage?.timeToReach ??
                        "3–4 months after fruit set",
                  ),
                  SizedBox(height: 8),
                  _infoRow(
                    Icons.hourglass_bottom,
                    "Time to Next Stage",
                    widget.maturityStage?.timeGapToNextStage ?? "3–7 days",
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            Material(
              color: const Color(0xFFEFF6FF),
              borderRadius: BorderRadius.circular(12),
              child: InkWell(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => MaturityGuideScreen()),
                  );
                },
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: const [
                      Text(
                        "Learn About Maturity Levels",
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFF2563EB),
                        ),
                      ),
                      SizedBox(width: 8),
                      Icon(
                        Icons.arrow_forward,
                        size: 20,
                        color: Color(0xFF2563EB),
                      ),
                    ],
                  ),
                ),
              ),
            ),

            SizedBox(height: 12),

            // Save button
            OutlinedButton(
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black87,
                minimumSize: Size(double.infinity, 50),
                side: BorderSide(color: Colors.grey[300]!),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () {},
              child: Text(
                "Save to Favorites",
                style: TextStyle(fontSize: 16),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoRow(IconData icon, String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: Colors.grey[600]),
        SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[700],
                ),
              ),
              SizedBox(height: 2),
              Text(
                value,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
