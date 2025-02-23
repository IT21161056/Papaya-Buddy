import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/login_view.dart';
import 'package:mobile_app/views/diagonosisView/diagnosis_list.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';
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
    const Center(child: Text('Disease View', style: TextStyle(fontSize: 20))),
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
      appBar: AppBar(title: const Text('Cure Your Crop')),
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
          Expanded(
            flex: 2,
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 10,
                mainAxisSpacing: 10,
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
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => DiseaseView()),
              );
            },
            child: const Text("tempory"),
          ),
        ],
      ),
    );
  }
}

// Crop Card Widget
class CropCard extends StatelessWidget {
  final String cropName;
  final IconData icon;
  final VoidCallback onTap;

  const CropCard({
    super.key,
    required this.cropName,
    required this.icon,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Card(
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 50, color: Colors.green),
            const SizedBox(height: 10),
            Center(
              child: Text(
                cropName,
                style: const TextStyle(
                    fontSize: 18, fontWeight: FontWeight.normal),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
