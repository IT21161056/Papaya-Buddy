import 'package:flutter/material.dart';
import 'package:carousel_slider/carousel_slider.dart';
import 'package:mobile_app/models/diseaseModel.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/views/treatment_view.dart';

class DiseaseView extends StatefulWidget {
  final DiseaseDisplayModel? disease;

  DiseaseView({Key? key, this.disease}) : super(key: key);

  @override
  _DiseaseViewState createState() => _DiseaseViewState();
}

class _DiseaseViewState extends State<DiseaseView> {
  bool _isSymptomExpanded = false;
  bool _isCauseExpanded = false;
  int _currentImageIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Diagnosis"),
        leading: IconButton(
          icon: Icon(Icons.arrow_back_ios, color: Colors.black),
          iconSize: 16,
          alignment: Alignment.center,
          onPressed: () {
            Navigator.pop(context);
          },
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.share),
            onPressed: () {},
          ),
          IconButton(
            icon: Icon(Icons.more_vert),
            onPressed: () {},
          ),
        ],
      ),
      backgroundColor: Color(0xFFF8FAFC),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
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
                        widget.disease?.name ?? 'Ring Spot Virus',
                        style: TextStyle(
                            fontSize: 22, fontWeight: FontWeight.bold),
                      ),
                      SizedBox(height: 4),
                      Text(
                        widget.disease?.diseaseType ?? 'Virus',
                        style: TextStyle(fontSize: 16, color: Colors.green),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.bug_report,
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
                  items: widget.disease?.suggestedImageUrls.map((path) {
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
                      "${_currentImageIndex + 1}/${widget.disease?.suggestedImageUrls.length ?? 0} photos",
                      style: TextStyle(fontSize: 12),
                    ),
                  ),
                ),
              ],
            ),

            SizedBox(height: 16),

            // Information box
            Container(
              padding: EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Icon(Icons.info_outline, color: Colors.blue),
                  SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      widget.disease?.description ??
                          'Caused by a single-stranded RNA virus belonging to the Potyvirus genus in the family Potyviridae.',
                      style: TextStyle(color: Colors.blue[900]),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // What caused it section
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey[300]!),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "What caused it?",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 12),
                  Row(
                    children: [
                      // ClipRRect(
                      //   borderRadius: BorderRadius.circular(8),
                      //   child: Image.asset(
                      //     'assets/r2.jpg',
                      //     width: 60,
                      //     height: 60,
                      //     fit: BoxFit.cover,
                      //   ),
                      // ),
                      SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            "Aphids",
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(
                            "Insect",
                            style: TextStyle(fontSize: 12),
                          ),
                        ],
                      ),
                    ],
                  ),
                  if (_isCauseExpanded) ...[
                    SizedBox(height: 12),
                    Text(
                      "Aphids are small sap-sucking insects that can transmit the Ring Spot Virus when feeding on plants. They pierce plant tissues with their needle-like mouthparts and inject the virus into the plant's vascular system. Once infected, the virus moves systemically throughout the plant, affecting fruits and foliage.",
                      style: TextStyle(fontSize: 14),
                    ),
                  ],
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: () {
                        setState(() {
                          _isCauseExpanded = !_isCauseExpanded;
                        });
                      },
                      child: Text(
                        _isCauseExpanded
                            ? "Show less"
                            : "See more about the cause",
                        style: TextStyle(color: Colors.indigo),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // Symptoms section
            Container(
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey[300]!),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Symptoms",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 12),
                  _symptomItem("Dark Green Rings on fruits"),
                  _symptomItem(
                      "Uneven yellowing or green mottling on the fruit."),
                  _symptomItem(
                      "The affected fruit may develop a rough, bumpy, or uneven surface."),
                  _symptomItem("The skin becomes harder than usual."),
                  if (_isSymptomExpanded) ...[
                    _symptomItem("Reduced fruit size and quality"),
                    _symptomItem("Premature fruit drop in severe cases"),
                    _symptomItem(
                        "Leaf symptoms may include vein clearing and mild mosaic patterns"),
                  ],
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: () {
                        setState(() {
                          _isSymptomExpanded = !_isSymptomExpanded;
                        });
                      },
                      child: Text(
                        _isSymptomExpanded ? "Show less" : "See more",
                        style: TextStyle(color: Colors.indigo),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            SizedBox(height: 16),

            // Treatment button
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                foregroundColor: Colors.white,
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => TreatmentScreen()),
                );
              },
              child: Text(
                "Treatment Instructions",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
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
                "Save to Diagnoses",
                style: TextStyle(fontSize: 16),
              ),
            ),
          ],
        ),
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
            margin: EdgeInsets.only(top: 6, right: 8),
            width: 6,
            height: 6,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.black87,
            ),
          ),
          Expanded(
            child: Text(text),
          ),
        ],
      ),
    );
  }
}
