import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/login_view.dart';
import 'package:mobile_app/views/auth/profile_view.dart';
import 'package:mobile_app/views/auth/signup_view.dart';
import 'package:mobile_app/views/dashboard_view.dart';
import 'package:mobile_app/views/diseaseView/disease_view.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  final List<Widget> _pages = [
    const DashboardView(),
    DiseaseView(),
    FirebaseAuth.instance.currentUser == null
        ? const LoginView()
        : const ProfileScreen(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex], // Display the selected page
      bottomNavigationBar: BottomNavigationBar(
        backgroundColor: Colors.white,
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        items: const [
          BottomNavigationBarItem(
              icon: Icon(
                Icons.home,
                color: Color.fromRGBO(100, 116, 139, 1),
              ),
              label: 'Home'),
          BottomNavigationBarItem(
            icon: Icon(
              Icons.comment,
              color: Color.fromRGBO(100, 116, 139, 1),
            ),
            label: ('Community'),
          ),
          BottomNavigationBarItem(
              icon: Icon(
                Icons.person,
                color: Color.fromRGBO(100, 116, 139, 1),
              ),
              label: 'Me'),
        ],
      ),
    );
  }
}
