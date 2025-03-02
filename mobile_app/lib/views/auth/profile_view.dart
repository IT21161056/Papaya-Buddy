import 'package:flutter/material.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  // Example user data - in a real app, this would come from your user repository or API
  final Map<String, dynamic> userData = {
    'name': 'John Doe',
    'email': 'johndoe@gmail.com',
    'joinDate': 'February 2025',
    'imageUrl': '', // Empty for now, we'll use a placeholder
  };

  // Controllers for editable fields
  late TextEditingController _nameController;
  late TextEditingController _emailController;

  // State variables
  bool _isEditing = false;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: userData['name']);
    _emailController = TextEditingController(text: userData['email']);
  }

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  // Save profile changes
  void _saveChanges() {
    setState(() {
      userData['name'] = _nameController.text;
      userData['email'] = _emailController.text;
      _isEditing = false;
    });

    // Here you would typically make an API call to update the user profile
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Profile updated successfully')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFEEF1F8),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Color(0xFF3F51B5)),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Profile',
          style: TextStyle(
            color: Color(0xFF3F51B5),
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(
              _isEditing ? Icons.save : Icons.edit,
              color: const Color(0xFF3F51B5),
            ),
            onPressed: () {
              if (_isEditing) {
                _saveChanges();
              } else {
                setState(() {
                  _isEditing = true;
                });
              }
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // Profile card with image and name
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    // Profile image
                    Stack(
                      alignment: Alignment.bottomRight,
                      children: [
                        CircleAvatar(
                          radius: 50,
                          backgroundColor:
                              const Color(0xFF3F51B5).withOpacity(0.1),
                          child: const Icon(
                            Icons.person,
                            size: 60,
                            color: Color(0xFF3F51B5),
                          ),
                        ),
                        if (_isEditing)
                          Container(
                            padding: const EdgeInsets.all(4),
                            decoration: const BoxDecoration(
                              color: Color(0xFF3F51B5),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(
                              Icons.camera_alt,
                              color: Colors.white,
                              size: 18,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // User info
                    _isEditing
                        ? _buildEditableField(
                            controller: _nameController,
                            label: 'Full Name',
                            icon: Icons.person_outline,
                          )
                        : Text(
                            userData['name'],
                            style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF333333),
                            ),
                          ),
                    const SizedBox(height: 4),
                    if (!_isEditing)
                      Text(
                        userData['email'],
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey[600],
                        ),
                      ),
                    if (!_isEditing) const SizedBox(height: 16),
                    if (!_isEditing)
                      Text(
                        'Member since ${userData['joinDate']}',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey[500],
                        ),
                      ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Profile details section
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Account Details',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF333333),
                      ),
                    ),
                    const SizedBox(height: 16),
                    if (_isEditing) ...[
                      _buildEditableField(
                        controller: _emailController,
                        label: 'Email Address',
                        icon: Icons.email_outlined,
                        isEmail: true,
                      ),
                      const SizedBox(height: 16),
                    ] else ...[
                      _buildInfoRow(
                        icon: Icons.email_outlined,
                        title: 'Email Address',
                        value: userData['email'],
                      ),
                      const Divider(height: 32),
                    ],
                    _buildInfoRow(
                      icon: Icons.calendar_today_outlined,
                      title: 'Member Since',
                      value: userData['joinDate'],
                    ),
                    const Divider(height: 32),
                    _buildSettingsButton(
                      icon: Icons.lock_outlined,
                      title: 'Change Password',
                      onTap: () {
                        // Navigate to change password screen
                      },
                    ),
                    const Divider(height: 32),
                    _buildSettingsButton(
                      icon: Icons.notifications_outlined,
                      title: 'Notification Settings',
                      onTap: () {
                        // Navigate to notification settings
                      },
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Action buttons section
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'App Settings',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF333333),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _buildSettingsButton(
                      icon: Icons.language_outlined,
                      title: 'Language',
                      value: 'English',
                      onTap: () {
                        // Navigate to language selection
                      },
                    ),
                    const Divider(height: 32),
                    _buildSettingsButton(
                      icon: Icons.dark_mode_outlined,
                      title: 'Dark Mode',
                      isSwitch: true,
                      onTap: () {
                        // Toggle dark mode
                      },
                    ),
                    const Divider(height: 32),
                    _buildSettingsButton(
                      icon: Icons.help_outline,
                      title: 'Help & Support',
                      onTap: () {
                        // Navigate to help section
                      },
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Logout button
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 10,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: TextButton.icon(
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.all(16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  icon: const Icon(
                    Icons.logout,
                    color: Colors.red,
                  ),
                  label: const Text(
                    'Logout',
                    style: TextStyle(
                      color: Colors.red,
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  onPressed: () {
                    // Logout functionality
                    showDialog(
                      context: context,
                      builder: (context) => AlertDialog(
                        title: const Text('Logout'),
                        content: const Text('Are you sure you want to logout?'),
                        actions: [
                          TextButton(
                            onPressed: () => Navigator.pop(context),
                            child: const Text('Cancel'),
                          ),
                          TextButton(
                            onPressed: () {
                              // Handle logout logic
                              Navigator.pop(context);
                              // Navigate to login screen
                              Navigator.of(context)
                                  .pushReplacementNamed('/login');
                            },
                            child: const Text(
                              'Logout',
                              style: TextStyle(color: Colors.red),
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
              ),

              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  // Helper widget to display info row
  Widget _buildInfoRow({
    required IconData icon,
    required String title,
    required String value,
  }) {
    return Row(
      children: [
        Icon(
          icon,
          size: 22,
          color: const Color(0xFF3F51B5),
        ),
        const SizedBox(width: 16),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: Color(0xFF333333),
              ),
            ),
          ],
        ),
      ],
    );
  }

  // Helper widget for settings button
  Widget _buildSettingsButton({
    required IconData icon,
    required String title,
    String? value,
    bool isSwitch = false,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          children: [
            Icon(
              icon,
              size: 22,
              color: const Color(0xFF3F51B5),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: Color(0xFF333333),
                ),
              ),
            ),
            if (value != null && !isSwitch)
              Text(
                value,
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.grey[600],
                ),
              ),
            if (isSwitch)
              Switch(
                value: false, // Would be connected to a state variable
                activeColor: const Color(0xFF3F51B5),
                onChanged: (val) => onTap(),
              ),
            if (!isSwitch && value == null)
              const Icon(
                Icons.arrow_forward_ios,
                size: 16,
                color: Colors.grey,
              ),
          ],
        ),
      ),
    );
  }

  // Helper widget for editable fields
  Widget _buildEditableField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    bool isEmail = false,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(icon, color: Colors.grey),
          const SizedBox(width: 10),
          Expanded(
            child: TextField(
              controller: controller,
              decoration: InputDecoration(
                labelText: label,
                border: InputBorder.none,
                labelStyle: const TextStyle(color: Colors.grey),
              ),
              keyboardType:
                  isEmail ? TextInputType.emailAddress : TextInputType.text,
            ),
          ),
        ],
      ),
    );
  }
}
