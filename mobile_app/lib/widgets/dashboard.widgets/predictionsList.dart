import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:mobile_app/models/predictionModel.dart';
import 'package:mobile_app/services/predictionService.dart';
import 'package:mobile_app/views/auth/login_view.dart';
import 'package:mobile_app/widgets/dashboard.widgets/noPredictions.dart';

import 'package:mobile_app/widgets/dashboard.widgets/predictionListItem.dart';
import 'package:mobile_app/widgets/dashboard.widgets/userNotLogged.dart';

class PredictionsList extends StatefulWidget {
  final String? userId;
  final double height;
  final Function(bool)? onLoadingChanged;
  final VoidCallback? onLoginPressed;
  final VoidCallback? onScanPressed;

  const PredictionsList({
    Key? key,
    this.userId,
    this.height = 300,
    this.onLoadingChanged,
    this.onLoginPressed,
    this.onScanPressed,
  }) : super(key: key);

  @override
  _PredictionsListState createState() => _PredictionsListState();
}

class _PredictionsListState extends State<PredictionsList> {
  List<Prediction> predictions = [];
  bool isLoading = true;
  String errorMessage = '';
  bool get isUserLoggedIn => widget.userId != null && widget.userId!.isNotEmpty;

  @override
  void initState() {
    super.initState();
    if (isUserLoggedIn) {
      _loadPredictions();
    } else {
      setState(() {
        isLoading = false;
      });
      widget.onLoadingChanged?.call(false);
    }
  }

  @override
  void didUpdateWidget(PredictionsList oldWidget) {
    super.didUpdateWidget(oldWidget);

    if (widget.userId != oldWidget.userId && isUserLoggedIn) {
      _loadPredictions();
    }
  }

  Future<void> _loadPredictions() async {
    setState(() {
      isLoading = true;
      errorMessage = '';
    });

    widget.onLoadingChanged?.call(true);

    try {
      List<Prediction> fetchedPredictions = await HistoryService.getDiagnosis(
        userId: widget.userId!,
      );

      if (mounted) {
        setState(() {
          predictions = fetchedPredictions;
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          errorMessage = 'Failed to load predictions';
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: _buildContent(),
    );
  }

  Widget _buildContent() {
    // Check if user is logged in
    if (!isUserLoggedIn) {
      return UserNotLoggedInView(
          onLoginPressed: () => {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => LoginView()),
                )
              });
    }

    if (isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (errorMessage.isNotEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(errorMessage, style: TextStyle(color: Colors.red[700])),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadPredictions,
              child: const Text('Try Again'),
            ),
          ],
        ),
      );
    }

    if (predictions.isEmpty) {
      return NoPredictionsView(onScanPressed: widget.onScanPressed);
    }

    return RefreshIndicator(
      onRefresh: _loadPredictions,
      child: ListView.separated(
        padding: EdgeInsets.zero,
        itemCount: predictions.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final prediction = predictions[index];
          return _buildPredictionListItem(prediction);
        },
      ),
    );
  }

  Widget _buildPredictionListItem(Prediction prediction) {
    final dateFormat = DateFormat('MMM d, yyyy');
    final image_url = prediction.uploadedImgUrl;
    final formattedDate =
        dateFormat.format(DateTime.parse(prediction.createdAt));

    final diseaseName = prediction.disease?.name ?? 'Unknown';
    final isHealthy = diseaseName.toLowerCase().contains('healthy');
    final resultStatus = isHealthy ? 'Healthy' : 'Unhealthy';

    return PredictionListItem(
      title: diseaseName,
      date: formattedDate,
      result: resultStatus,
      image: image_url,
      onDetailsPressed: () {
        // Handle details action
      },
    );
  }
}
