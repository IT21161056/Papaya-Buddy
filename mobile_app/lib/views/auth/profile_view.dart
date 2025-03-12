import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/widgets/profile.widgets/accout_settings_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/contact_info_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/logout_alert_dialog.dart';
import 'package:mobile_app/widgets/profile.widgets/profile_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/subscription_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/support_widget.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  AuthService _authService = new AuthService();
  bool notificationsEnabled = true;
  bool locationEnabled = true;
  bool analyticsEnabled = true;
  double logoutScale = 1.0;
  bool _isLoading = false;

  Timer? _logoutTimeout;

  void handleLogout() async {
    bool confirmLogout = await showDialog(
      context: context,
      builder: (context) => LogoutDialog(),
    );

    if (confirmLogout == true) {
      setState(() => _isLoading = true);

      try {
        await _authService.signOut();

        if (mounted) {
          setState(() => _isLoading = false);
          Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
        }
      } catch (e) {
        if (mounted) {
          setState(() => _isLoading = false);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Error: ${e.toString()}")),
          );
        }
      }
    }
  }

  void animateLogout(bool isPressed) {
    setState(() {
      logoutScale = isPressed ? 0.95 : 1.0;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text(
          "Profile",
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        centerTitle: false,
        backgroundColor: Colors.white,
        elevation: 1,
        actions: [
          IconButton(
            onPressed: () {},
            icon: SvgPicture.asset(
              'assets/icons/lucide_bell.svg',
              height: 22,
              color: Colors.black,
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            ProfileCard(
                userName: "Alex Johnson",
                description: 'Plant enthusiast & organic farmer',
                profilePicture:
                    'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?q=80'),
            const SizedBox(height: 24),
            ContactInfoCard(
              email: "alex.johnson@example.com",
              phone: "+1 (555) 123-4567",
              address: "Portland, Oregon",
            ),
            const SizedBox(height: 24),
            AccountSettingsCard(),
            const SizedBox(height: 24),
            SubscriptionPlanCard(
              currentPlan: "Premium Plan",
              description:
                  'Access to all premium features including unlimited plant scans and detailed treatment plans.',
              renewalDate: 'June 15, 2025',
              plan: "Annual (\$59.99/year)",
              active: true,
              onTap: () {},
            ),

            const SizedBox(height: 24),
            SupportCard(),

            const SizedBox(height: 16),
            // Logout Button
            SizedBox(
              width: double.infinity,
              child: TextButton(
                onPressed: handleLogout,
                style: TextButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 24),
                  backgroundColor: const Color(0xFFFEF2F2),
                  foregroundColor: const Color(0xFFEF4444),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.logout,
                      color: Colors.red,
                    ),
                    SizedBox(width: 8),
                    Text(
                      "Log Out",
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),
            const Text(
              "PapayaBuddy v1.0.0",
              style: TextStyle(color: Color(0xFF94A3B8)),
            ),
          ],
        ),
      ),
    );
  }
}
