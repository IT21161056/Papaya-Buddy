import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/login_view.dart';
import 'package:mobile_app/views/diagonosisView/diagnosis_list.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
import 'package:mobile_app/views/treatment_view.dart';
import 'package:mobile_app/widgets/home.widgets/homeWidget.dart';
import 'package:mobile_app/widgets/imagePickerWidget/fruitdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/imagepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/leafdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/maturitypicker.dart';
import 'package:mobile_app/widgets/whether-widgets/whetherWidget.dart';

class HomeView extends StatefulWidget {
  const HomeView({super.key});

  @override
  _HomeViewState createState() => _HomeViewState();
}

// Crop Cards Data
final List<Map<String, dynamic>> cropCards = [
  {
    'name': 'Fruit Disease Detection',
    'icon': Icons.bug_report,
    'picker': FruitDiseasePicker()
  },
  {
    'name': 'Leaf Disease Detection',
    'icon': Icons.grass,
    'picker': LeafDiseasePicker()
  },
  {
    'name': 'Maturity Level',
    'icon': Icons.agriculture,
    'picker': MaturityPicker()
  },
  {
    'name': 'Pest Detection',
    'icon': Icons.bar_chart,
    'picker': ImagePickerPage()
  },
];

class _HomeViewState extends State<HomeView> {
  int _selectedIndex = 0;

  static final List<Widget> _pages = <Widget>[
    const Dashboard(showImage: true),
    DiseaseView(),
    const LoginView(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFFF8FAFC),
      appBar: AppBar(
          automaticallyImplyLeading: false, backgroundColor: Colors.white),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(
              icon: Icon(Icons.local_hospital), label: 'Diseases'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
        ],
        currentIndex: _selectedIndex,
        selectedItemColor: Colors.green,
        onTap: _onItemTapped,
      ),
    );
  }
}

// Dashboard UI
class Dashboard extends StatelessWidget {
  final bool showImage;
  const Dashboard({super.key, required this.showImage});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (showImage)
            Align(
              alignment: Alignment.topLeft,
              child: Padding(
                  padding: const EdgeInsets.only(left: 2.0),
                  child: const WeatherWidget()),
            ),
          const Padding(
            padding: EdgeInsets.only(top: 0.0, bottom: 5.0),
            child: Text(
              'Cure Your Crop',
              style: TextStyle(
                fontSize: 21,
                fontWeight: FontWeight.bold,
                color: Color.fromARGB(255, 9, 11, 9),
              ),
            ),
          ),
          SizedBox(
            height: 10,
          ),
          Expanded(
            flex: 2,
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 5,
                mainAxisSpacing: 10,
                mainAxisExtent: 150,
              ),
              itemCount: cropCards.length,
              itemBuilder: (context, index) {
                return CropCard(
                  cropName: cropCards[index]['name'],
                  icon: cropCards[index]['icon'],
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => cropCards[index]['picker'],
                      ),
                    );
                  },
                );
              },
            ),
          ),
          const SizedBox(height: 16),
          const Expanded(
            flex: 1,
            child: DiagnosisList(),
          ),
        ],
      ),
    );
  }
}
