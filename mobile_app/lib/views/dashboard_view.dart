import 'package:flutter/material.dart';
import 'package:mobile_app/views/diagonosisView/diagnosis_list.dart';
import 'package:mobile_app/views/diagonosisView/disease_details.dart';
import 'package:mobile_app/widgets/dashboard.widgets/crop_card.dart';
import 'package:mobile_app/widgets/dashboard.widgets/weather_widget.dart';
import 'package:mobile_app/widgets/imagePickerWidget/fruitdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/imagepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/leafdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/maturitypicker.dart';

class DashboardView extends StatefulWidget {
  const DashboardView({super.key});
  @override
  State<DashboardView> createState() => _DashboardViewState();
}

// Crop Cards Data
final List<Map<String, dynamic>> cropCards = [
  {
    'name': 'Fruit Disease',
    'icon': 'assets/icons/carbon_fruit_bowl.svg',
    'picker': FruitDiseasePicker(),
    'color': Colors.orange
  },
  {
    'name': 'Leaf Disease',
    'icon': 'assets/icons/lucide_leaf.svg',
    'picker': LeafDiseasePicker(),
    'color': Colors.green
  },
  {
    'name': 'Maturity Level',
    'icon': 'assets/icons/lucide_chart.svg',
    'picker': MaturityPicker(),
    'color': Colors.blue
  },
  {
    'name': 'Pest Detection',
    'icon': 'assets/icons/material_pest.svg',
    'picker': ImagePickerPage(),
    'color': Colors.blueGrey
  },
];

class _DashboardViewState extends State<DashboardView> {
  bool isExpanded = false;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          centerTitle: false,
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
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Welcome back,",
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  color: Color.fromRGBO(100, 116, 139, 1),
                  fontSize: 14,
                ),
              ),
              Text(
                "Alex",
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  color: Colors.black,
                  fontSize: 20,
                ),
              ),
            ],
          ),
        ),
        backgroundColor: Color(0xFFF8FAFC),
        body: SafeArea(
            child: Column(
          children: [
            // Wether widget
            ExpandableWeatherCard(),
            // "Cure Your Crop" Section
            Expanded(
                child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        "Cure Your Crop",
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      // Text(
                      //   "See all",
                      //   style: TextStyle(
                      //       fontSize: 16,
                      //       color: Colors.blue,
                      //       fontWeight: FontWeight.bold),
                      // ),
                    ],
                  ),
                  const SizedBox(
                    height: 12,
                  ),
                  ConstrainedBox(
                    constraints: const BoxConstraints(
                      minHeight: 0,
                    ),
                    child: GridView.builder(
                      physics: const NeverScrollableScrollPhysics(),
                      shrinkWrap: true,
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        mainAxisSpacing: 12,
                        crossAxisSpacing: 12,
                        childAspectRatio: 1.5, //2, 1.8
                      ),
                      itemCount: cropCards.length,
                      itemBuilder: (context, index) {
                        return SizedBox(
                          height: 100,
                          child: CropCard(
                            title: cropCards[index]['name'],
                            iconPath: cropCards[index]['icon'],
                            color: cropCards[index]['color'],
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      cropCards[index]['picker'],
                                ),
                              );
                            },
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(
                    height: 24,
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        "Your Diagnoses",
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold),
                      ),
                      // Text(
                      //   "See all",
                      //   style: TextStyle(
                      //       fontSize: 16,
                      //       color: Colors.blue,
                      //       fontWeight: FontWeight.bold),
                      // ),
                    ],
                  ),
                  const SizedBox(
                    height: 12,
                  ),
                  ListView.separated(
                    physics: const NeverScrollableScrollPhysics(),
                    shrinkWrap: true,
                    itemCount: pastDiagnoses.length,
                    separatorBuilder: (context, index) => const SizedBox(
                      height: 10,
                    ),
                    itemBuilder: (context, index) {
                      return DiagnosisListItem(
                        title: pastDiagnoses[index]['title'],
                        date: pastDiagnoses[index]['date'],
                        result: pastDiagnoses[index]['result'],
                        onDetailsPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => DiseaseDetailsPage(
                                diseaseName: pastDiagnoses[index]['title'] ??
                                    'No Title Available',
                                description: pastDiagnoses[index]
                                        ['description'] ??
                                    'No description available',
                                remedy: pastDiagnoses[index]['remedy'] ??
                                    'No remedy available',
                                images: pastDiagnoses[index]['images'] ?? [],
                              ),
                            ),
                          );
                        },
                      );
                    },
                  ),
                ],
              ),
            )),
          ],
        )));
  }
}
