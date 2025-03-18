import 'package:flutter/material.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/widgets/dashboard.widgets/crop_card.dart';
import 'package:mobile_app/widgets/dashboard.widgets/predictionsList.dart';
import 'package:mobile_app/widgets/weather.widgets/weather_widget.dart';
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
    'picker': MaturityScreen(),
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
  final AuthService _authService = AuthService();
  bool _isLoading = true;
  bool isExpanded = false;
  bool isPredictionsLoading = false;
  String? userUID;

  @override
  void initState() {
    super.initState();
    loadUserData();
  }

  Future<void> loadUserData() async {
    userUID = _authService.getUserUID();
    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Color(0xFFF8FAFC),
        body: SafeArea(
            child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(
                  bottom: BorderSide(
                    color: Color(0xFFF1F5F9),
                    width: 1,
                  ),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
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
                        childAspectRatio: 1.6, //2, 1.8
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

                  // Old Prediction List
                  PredictionsList(
                    userId: userUID ?? '',
                    height: 300,
                    onLoadingChanged: (isLoading) {},
                  ),
                ],
              ),
            )),
          ],
        )));
  }
}
