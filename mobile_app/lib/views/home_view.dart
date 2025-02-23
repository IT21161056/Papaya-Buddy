import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/login_view.dart';
import 'package:mobile_app/views/diagonosisView/diagnosis_list.dart';
import 'package:mobile_app/views/treatment_view.dart';
import 'package:mobile_app/widgets/home.widgets/homeWidget.dart';
import 'package:mobile_app/widgets/imagePickerWidget/fruitdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/imagepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/leafdiseasepicker.dart';
import 'package:mobile_app/widgets/imagePickerWidget/maturitypicker.dart';

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
    'name': 'Fruit/Leaf Detection',
    'icon': Icons.bar_chart,
    'picker': ImagePickerPage()
  },
];

class _HomeViewState extends State<HomeView> {
  int _selectedIndex = 0;

  static final List<Widget> _pages = <Widget>[
    const Dashboard(),
    TreatmentScreen(),
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
      appBar: AppBar(
        automaticallyImplyLeading: false, // Removes the leading button (back)
        flexibleSpace: Padding(
          padding: const EdgeInsets.only(top: 29.0), // Adds a top margin
          child: Align(
            alignment: Alignment.topLeft,
            child: Image.asset(
              'assets/papaya.png', // Image location
              width: 80, // Increased width size
              height: 90, // Increased height size
            ),
          ),
        ),
      ),
      body: _pages[_selectedIndex], // Show selected page
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
  const Dashboard({super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        children: [
          // Add "Cure Your Crop" title above the cards
          const Padding(
            padding: EdgeInsets.only(bottom: 16.0),
            child: Text(
              'Cure Your Crop',
              style: TextStyle(
                fontSize: 21,
                fontWeight: FontWeight.bold,
                color: Color.fromARGB(255, 9, 11, 9),
              ),
            ),
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
