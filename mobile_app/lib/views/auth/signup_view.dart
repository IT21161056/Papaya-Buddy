import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class SignupView extends StatefulWidget {
  const SignupView({super.key});
  @override
  _SignupViewState createState() => _SignupViewState();
}

class _SignupViewState extends State<SignupView> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController =
      TextEditingController();

  bool _isPasswordVisible = false;
  bool _isConfirmPasswordVisible = false;
  void _signUp() {
    if (_formKey.currentState!.validate()) {
      // Proceed with signup logic
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Signup Successful!')),
      );
      Navigator.pushReplacementNamed(context, '/home');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
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
          "Create Account",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: Colors.black,
            fontSize: 20,
          ),
        ),
      ),
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Center(
              child: Column(
                children: [
                  Text(
                    "Join PapayaBuddy to diagnose and treat plant diseases",
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),
            TextField(
              controller: _nameController,
              decoration: InputDecoration(
                prefixIcon: Padding(
                  padding: const EdgeInsets.all(
                      12), // Adjust padding for better alignment
                  child: SizedBox(
                    height: 18, // Adjust icon size
                    width: 18,
                    child: SvgPicture.asset(
                      'assets/icons/user.svg', // Replace with correct icon
                      height: 12,
                      width: 12,
                      color: Color(0xFF64748B),
                    ),
                  ),
                ),
                hintText: "Full Name",
                filled: true,
                fillColor: Color(0xFFF8FAFC),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0)), // Border color added
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0),
                      width: 2), // Slightly thicker on focus
                ),
              ),
            ),
            const SizedBox(height: 15),
            TextField(
              controller: _emailController,
              decoration: InputDecoration(
                prefixIcon: Padding(
                  padding: const EdgeInsets.all(
                      12), // Adjust padding for better alignment
                  child: SizedBox(
                    height: 18, // Adjust icon size
                    width: 18,
                    child: SvgPicture.asset(
                      'assets/icons/lucide_mail.svg', // Replace with correct icon
                      height: 12,
                      width: 12,
                      color: Color(0xFF64748B),
                    ),
                  ),
                ),
                hintText: "Email",
                filled: true,
                fillColor: Color(0xFFF8FAFC),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0)), // Border color added
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0),
                      width: 2), // Slightly thicker on focus
                ),
              ),
            ),

            const SizedBox(height: 15),

            // Password
            TextField(
              controller: _passwordController,
              obscureText: !_isPasswordVisible,
              decoration: InputDecoration(
                prefixIcon: Padding(
                  padding: const EdgeInsets.all(
                      12), // Adjust padding for better alignment
                  child: SizedBox(
                    height: 18, // Adjust icon size
                    width: 18,
                    child: SvgPicture.asset(
                      'assets/icons/lucide_lock.svg', // Replace with correct icon
                      height: 12,
                      width: 12,
                      color: Color(0xFF64748B),
                    ),
                  ),
                ), // Using exact RGBO equivalent
                suffixIcon: IconButton(
                  icon: Icon(
                    _isPasswordVisible
                        ? Icons.visibility
                        : Icons.visibility_off,
                    color: Color(0xFF64748B),
                  ),
                  onPressed: () {
                    setState(() {
                      _isPasswordVisible = !_isPasswordVisible;
                    });
                  },
                ),
                hintText: "Password",
                filled: true,
                fillColor: Color(0xFFF8FAFC),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0)), // Light border
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0),
                      width: 2), // Thicker when focused
                ),
              ),
            ),

            const SizedBox(height: 15),

            // Confirm Password
            TextField(
              controller: _confirmPasswordController,
              obscureText: !_isConfirmPasswordVisible,
              decoration: InputDecoration(
                prefixIcon: Padding(
                  padding: const EdgeInsets.all(
                      12), // Adjust padding for better alignment
                  child: SizedBox(
                    height: 18, // Adjust icon size
                    width: 18,
                    child: SvgPicture.asset(
                      'assets/icons/lucide_lock.svg', // Replace with correct icon
                      height: 12,
                      width: 12,
                      color: Color(0xFF64748B),
                    ),
                  ),
                ), // Using exact RGBO equivalent
                suffixIcon: IconButton(
                  icon: Icon(
                    _isConfirmPasswordVisible
                        ? Icons.visibility
                        : Icons.visibility_off,
                    color: Color(0xFF64748B),
                  ),
                  onPressed: () {
                    setState(() {
                      _isConfirmPasswordVisible = !_isConfirmPasswordVisible;
                    });
                  },
                ),
                hintText: "Confirm Password",
                filled: true,
                fillColor: Color(0xFFF8FAFC),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0)), // Light border
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(
                      color: Color(0xFFE2E8F0),
                      width: 2), // Thicker when focused
                ),
              ),
            ),

            const SizedBox(height: 30),

            // button
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.blue,
                minimumSize: const Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              onPressed: _signUp,
              child: const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    "Create Account",
                    style: TextStyle(
                        fontSize: 16,
                        color: Colors.white,
                        fontWeight: FontWeight.w600),
                  ),
                  SizedBox(width: 8), // Space between text and icon
                  Icon(Icons.arrow_right_alt,
                      color: Colors.white), // Add forward icon
                ],
              ),
            ),

            const SizedBox(height: 15),
            Center(
              child: RichText(
                text: TextSpan(
                  text: "Already have an account? ",
                  style: const TextStyle(color: Colors.black),
                  children: [
                    WidgetSpan(
                      child: GestureDetector(
                        onTap: () {},
                        child: const Text(
                          "Sign In",
                          style: TextStyle(
                            color: Colors.blue,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
