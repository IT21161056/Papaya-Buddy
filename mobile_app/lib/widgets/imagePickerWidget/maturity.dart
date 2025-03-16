import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

class MaturityScreenNew extends StatefulWidget {
  const MaturityScreenNew({Key? key}) : super(key: key);

  @override
  State<MaturityScreenNew> createState() => _MaturityScreenState();
}

class _MaturityScreenState extends State<MaturityScreenNew> {
  String? selectedImage;
  Map<String, dynamic>? prediction;

  void handleSelectImage() {
    setState(() {
      selectedImage =
          'https://images.unsplash.com/photo-1587132137056-bfbf0166836e?q=80&w=500&auto=format&fit=crop';
    });
  }

  void handleAnalyze() {
    // Add your analysis logic here
    debugPrint('Analyzing image...');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      body: SafeArea(
        child: Column(
          children: [
            // Header
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(
                  bottom: BorderSide(
                    color: Color(0xFFF1F5F9),
                    width: 1,
                  ),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  InkWell(
                    onTap: () => Navigator.pop(context),
                    borderRadius: BorderRadius.circular(12),
                    child: Ink(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: const Color(0xFFF8FAFC),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(
                        Icons.chevron_left,
                        size: 24,
                        color: Color(0xFF1A1A1A),
                      ),
                    ),
                  ),
                  const Text(
                    'Maturity Detection',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF1A1A1A),
                    ),
                  ),
                  const SizedBox(width: 40),
                ],
              ),
            ),

            // Scrollable Content
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(20),
                child: Column(
                  children: [
                    // Main Card
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            offset: const Offset(0, 1),
                            blurRadius: 5,
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          // Image Container
                          AspectRatio(
                            aspectRatio: 4 / 3,
                            child: Container(
                              decoration: BoxDecoration(
                                color: const Color(0xFFF8FAFC),
                                borderRadius: BorderRadius.circular(16),
                              ),
                              child: selectedImage != null
                                  ? ClipRRect(
                                      borderRadius: BorderRadius.circular(16),
                                      child: Image.network(
                                        selectedImage!,
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      ),
                                    )
                                  : Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: const [
                                        Icon(
                                          Icons.document_scanner_outlined,
                                          size: 48,
                                          color: Color(0xFF94A3B8),
                                        ),
                                        SizedBox(height: 12),
                                        Text(
                                          'No image selected',
                                          style: TextStyle(
                                            fontSize: 16,
                                            fontWeight: FontWeight.w600,
                                            color: Color(0xFF64748B),
                                          ),
                                        ),
                                        SizedBox(height: 4),
                                        Text(
                                          'Take a photo or choose from gallery',
                                          style: TextStyle(
                                            fontSize: 14,
                                            color: Color(0xFF94A3B8),
                                          ),
                                          textAlign: TextAlign.center,
                                        ),
                                      ],
                                    ),
                            ),
                          ),
                          const SizedBox(height: 20),

                          // Action Buttons
                          Row(
                            children: [
                              Expanded(
                                child: _AnimatedButton(
                                  icon: Icons.camera_alt,
                                  label: 'Camera',
                                  color: const Color(0xFF2563EB),
                                  onPressed: () => debugPrint('Camera pressed'),
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: _AnimatedButton(
                                  icon: Icons.image,
                                  label: 'Gallery',
                                  color: const Color(0xFF2563EB),
                                  onPressed: handleSelectImage,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ).animate().fadeIn(delay: 200.ms),

                    const SizedBox(height: 24),

                    // Maturity Levels Card
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            offset: const Offset(0, 1),
                            blurRadius: 5,
                          ),
                        ],
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Ripeness Levels',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF1A1A1A),
                            ),
                          ),
                          const SizedBox(height: 16),

                          // Maturity Levels List
                          ...List.generate(
                            4,
                            (index) {
                              final levels = [
                                {
                                  'color': const Color(0xFF22C55E),
                                  'title': 'Ripe',
                                  'description':
                                      'Perfect for immediate consumption',
                                },
                                {
                                  'color': const Color(0xFF84CC16),
                                  'title': 'Partially Ripe',
                                  'description': 'Will be ready in 2-3 days',
                                },
                                {
                                  'color': const Color(0xFF0EA5E9),
                                  'title': 'Not Ripe',
                                  'description': 'Needs more time to ripen',
                                },
                                {
                                  'color': const Color(0xFFEF4444),
                                  'title': 'Overripe',
                                  'description':
                                      'Past optimal consumption time',
                                },
                              ];

                              return Padding(
                                padding: const EdgeInsets.only(bottom: 16),
                                child: InkWell(
                                  onTap: () => debugPrint(
                                      '${levels[index]['title']} pressed'),
                                  borderRadius: BorderRadius.circular(12),
                                  child: Ink(
                                    decoration: BoxDecoration(
                                      color: const Color(0xFFF8FAFC),
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    padding: const EdgeInsets.all(16),
                                    child: Row(
                                      children: [
                                        Container(
                                          width: 12,
                                          height: 48,
                                          decoration: BoxDecoration(
                                            color:
                                                levels[index]['color'] as Color,
                                            borderRadius:
                                                BorderRadius.circular(6),
                                          ),
                                        ),
                                        const SizedBox(width: 16),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                levels[index]['title']
                                                    as String,
                                                style: const TextStyle(
                                                  fontSize: 16,
                                                  fontWeight: FontWeight.w600,
                                                  color: Color(0xFF1A1A1A),
                                                ),
                                              ),
                                              const SizedBox(height: 4),
                                              Text(
                                                levels[index]['description']
                                                    as String,
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                  color: Color(0xFF64748B),
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              );
                            },
                          ),
                        ],
                      ),
                    )
                        .animate()
                        .fadeIn(delay: 300.ms)
                        .slideY(begin: 0.2, end: 0),

                    const SizedBox(height: 24),

                    // Tips Card
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.05),
                            offset: const Offset(0, 1),
                            blurRadius: 5,
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          Row(
                            children: const [
                              Icon(
                                Icons.info_outline,
                                size: 20,
                                color: Color(0xFF0284C7),
                              ),
                              SizedBox(width: 8),
                              Text(
                                'Tips for Better Results',
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.w600,
                                  color: Color(0xFF1A1A1A),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),

                          // Tips List
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF8FAFC),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Column(
                              children: [
                                ...List.generate(
                                  3,
                                  (index) {
                                    final tips = [
                                      'Ensure good lighting conditions',
                                      'Place fruit against a plain background',
                                      'Keep the camera steady and focused',
                                    ];

                                    return Padding(
                                      padding:
                                          const EdgeInsets.only(bottom: 12),
                                      child: InkWell(
                                        onTap: () => debugPrint(
                                            'Tip ${index + 1} pressed'),
                                        borderRadius: BorderRadius.circular(12),
                                        child: Row(
                                          children: [
                                            Container(
                                              width: 24,
                                              height: 24,
                                              decoration: BoxDecoration(
                                                color: const Color(0xFFE0F2FE),
                                                borderRadius:
                                                    BorderRadius.circular(12),
                                              ),
                                              alignment: Alignment.center,
                                              child: Text(
                                                '${index + 1}',
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                  fontWeight: FontWeight.w600,
                                                  color: Color(0xFF0284C7),
                                                ),
                                              ),
                                            ),
                                            const SizedBox(width: 12),
                                            Expanded(
                                              child: Text(
                                                tips[index],
                                                style: const TextStyle(
                                                  fontSize: 14,
                                                  color: Color(0xFF1A1A1A),
                                                  height: 1.4,
                                                ),
                                              ),
                                            ),
                                          ],
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    )
                        .animate()
                        .fadeIn(delay: 400.ms)
                        .slideY(begin: 0.2, end: 0),

                    const SizedBox(height: 24),

                    // Analyze Button (visible when image is selected)
                    if (selectedImage != null)
                      _AnimatedButton(
                        icon: Icons.arrow_forward,
                        label: 'Analyze Maturity',
                        color: const Color(0xFF16A34A),
                        onPressed: handleAnalyze,
                        fullWidth: true,
                      ).animate().fadeIn(delay: 500.ms),
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

class _AnimatedButton extends StatefulWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onPressed;
  final bool fullWidth;

  const _AnimatedButton({
    Key? key,
    required this.icon,
    required this.label,
    required this.color,
    required this.onPressed,
    this.fullWidth = false,
  }) : super(key: key);

  @override
  State<_AnimatedButton> createState() => _AnimatedButtonState();
}

class _AnimatedButtonState extends State<_AnimatedButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 150),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 0.95).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) => _controller.reverse(),
      onTapCancel: () => _controller.reverse(),
      onTap: widget.onPressed,
      child: AnimatedBuilder(
        animation: _scaleAnimation,
        builder: (context, child) {
          return Transform.scale(
            scale: _scaleAnimation.value,
            child: Container(
              width: widget.fullWidth ? double.infinity : null,
              padding: const EdgeInsets.symmetric(vertical: 16),
              decoration: BoxDecoration(
                color: widget.color,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    widget.label,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Icon(
                    widget.icon,
                    size: 20,
                    color: Colors.white,
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
