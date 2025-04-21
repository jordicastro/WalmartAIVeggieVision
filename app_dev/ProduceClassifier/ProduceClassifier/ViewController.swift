//
//  ViewController.swift
//  ProduceClassifier
//
//  Created by David Nystrom on 4/17/25.
//

import UIKit
import AVFoundation
import onnxruntime_objc

// MARK: — UIColor Helpers
// Extend UIColor to support hex-based initialization and brightness adjustment
extension UIColor {
    /// Initialize UIColor from a 0xRRGGBB hex value with optional alpha
    convenience init(hex: Int, alpha: CGFloat = 1.0) {
        // Extract red, green, blue components from the hex integer
        let r = CGFloat((hex >> 16) & 0xFF) / 255.0
        let g = CGFloat((hex >> 8) & 0xFF) / 255.0
        let b = CGFloat(hex & 0xFF) / 255.0
        self.init(red: r, green: g, blue: b, alpha: alpha)
    }

    /// Return a color lightened (positive factor) or darkened (negative)
    func adjustBrightness(by factor: CGFloat) -> UIColor {
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        // Retrieve current RGBA components; bail out if unavailable
        guard self.getRed(&r, green: &g, blue: &b, alpha: &a) else { return self }
        // Clamp each component to [0,1] after adjustment
        return UIColor(
            red:   min(max(r + factor, 0), 1),
            green: min(max(g + factor, 0), 1),
            blue:  min(max(b + factor, 0), 1),
            alpha: a
        )
    }
}

// MARK: — ViewController
// Main view controller handling camera capture, UI, and model inference
class ViewController: UIViewController {

    // MARK: – Capture
    // Capture session and preview layer for live camera feed
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue  = DispatchQueue(label: "videoQueue")

    // Flags to control inference flow
    private var isInferencing   = false  // User toggles inference on/off
    private var processingFrame = false  // Prevent overlapping frame processing

    // MARK: – UI
    private var toggleButton: UIButton!           // Button to start/stop inference
    private var inferenceTimeLabel: UILabel!      // Label to show inference time
    private var resultLabel: UILabel!             // Multi-line label for predictions

    /// Colors associated with each class for rendering results
    private let classColors: [String: UIColor] = [
        // Bagged items slightly darker to differentiate
        "bagged_banana":              UIColor(hex: 0xFDE74C).adjustBrightness(by: -0.1),
        "banana":                     UIColor(hex: 0xFFEB3B),
        "bagged_broccoli":            UIColor(hex: 0x4CAF50),
        "broccoli":                   UIColor(hex: 0x2E7D32),
        // Custom RGB for apple variants
        "bagged_fiji_apple":          UIColor(red: 239/255, green: 83/255, blue: 80/255, alpha: 1.0),
        "fiji_apple":                 UIColor(red: 244/255, green: 67/255, blue: 54/255, alpha: 1.0),
        "bagged_granny_apple":        UIColor(hex: 0x8BC34A),
        "granny_apple":               UIColor(hex: 0x558B2F),
        "bagged_jalapeno":            UIColor(hex: 0x689F38),
        "jalapeno":                   UIColor(hex: 0x33691E),
        "bagged_orange_bell_pepper":  UIColor(hex: 0xFF9800),
        "orange_bell_pepper":         UIColor(hex: 0xEF6C00),
        "bagged_red_bell_pepper":     UIColor(hex: 0xE53935),
        "red_bell_pepper":            UIColor(hex: 0xB71C1C)
    ]

    override func viewDidLoad() {
        super.viewDidLoad()
        // Set up camera feed and UI elements
        setupCamera()
        setupUI()
        // Initialize ONNX Runtime model; catch errors if loading fails
        do {
            try EfficientNetB0.initializeModel()
        } catch {
            print("❌ Model load failed:", error)
        }
    }

    // MARK: – Camera
    /// Configure AVCaptureSession, inputs, outputs, and preview
    private func setupCamera() {
        captureSession.sessionPreset = .high
        // Obtain default video capture device
        guard let device = AVCaptureDevice.default(for: .video),
              let input  = try? AVCaptureDeviceInput(device: device)
        else { fatalError("Camera unavailable") }
        captureSession.addInput(input)

        // Configure output to deliver CVPixelBuffer frames
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        captureSession.addOutput(videoOutput)

        // Add preview layer to view hierarchy for user feedback
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        // Start running camera on background thread to avoid UI block
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }

    // MARK: – UI
    /// Build and position UI controls: toggle button and labels
    private func setupUI() {
        // Toggle inference button
        toggleButton = UIButton(type: .system)
        toggleButton.frame = CGRect(
            x: (view.frame.width - 120) / 2,
            y: view.frame.height - 80,
            width: 120, height: 50
        )
        toggleButton.setTitle("Start", for: .normal)
        toggleButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        toggleButton.backgroundColor = .white
        toggleButton.layer.cornerRadius = 8
        toggleButton.addTarget(self, action: #selector(toggleInference), for: .touchUpInside)
        view.addSubview(toggleButton)

        // Label to display inference time in seconds
        inferenceTimeLabel = UILabel(frame: CGRect(x: 16, y: 40, width: 150, height: 30))
        inferenceTimeLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        inferenceTimeLabel.textColor = .white
        inferenceTimeLabel.font = .systemFont(ofSize: 14)
        inferenceTimeLabel.textAlignment = .center
        inferenceTimeLabel.text = "Time: --"
        view.addSubview(inferenceTimeLabel)

        // Multi-line label for top-5 classification results
        resultLabel = UILabel(frame: CGRect(
            x: 16, y: 80,
            width: view.frame.width - 32,
            height: 140
        ))
        resultLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        resultLabel.numberOfLines = 6
        resultLabel.layer.cornerRadius = 8
        resultLabel.clipsToBounds = true
        view.addSubview(resultLabel)
    }

    // MARK: – Toggle
    /// Toggle inference on/off and update UI button text
    @objc private func toggleInference() {
        isInferencing.toggle()
        DispatchQueue.main.async {
            let title = self.isInferencing ? "Stop" : "Start"
            self.toggleButton.setTitle(title, for: .normal)
            self.resultLabel.isHidden = !self.isInferencing
        }
    }
}

// MARK: – Video Output Delegate
// Handle each camera frame, run inference, and update UI
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Skip if not inferencing or already processing a frame
        guard isInferencing && !processingFrame else { return }
        processingFrame = true

        // Extract pixel buffer from sample buffer
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            processingFrame = false
            return
        }
        // Convert CVPixelBuffer to UIImage via CIImage
        let ciImage = CIImage(cvPixelBuffer: pb)
        let ctx     = CIContext()
        guard let cgImg = ctx.createCGImage(ciImage, from: ciImage.extent) else {
            processingFrame = false
            return
        }
        let uiImage = UIImage(cgImage: cgImg)

        let start = Date()
        // Perform inference off the main thread
        DispatchQueue.global(qos: .userInitiated).async {
            defer { self.processingFrame = false }
            do {
                // Get top-5 predictions from model
                let top5 = try EfficientNetB0.classify(image: uiImage)
                let elapsed = Date().timeIntervalSince(start)

                // Build attributed string to display time and results
                let attr = NSMutableAttributedString()
                attr.append(NSAttributedString(
                    string: String(format: "Time: %.2fs\n", elapsed),
                    attributes: [.foregroundColor: UIColor.white]
                ))

                // Append each label with its confidence, colored appropriately
                for (idx, prob) in top5 {
                    let label = EfficientNetB0.labelsMap[idx]!
                    let color = self.classColors[label] ?? .white
                    let line  = String(format: "%@: %.1f%%\n", label, prob * 100)
                    attr.append(NSAttributedString(
                        string: line,
                        attributes: [.foregroundColor: color]
                    ))
                }

                // Update UI with results and timing
                DispatchQueue.main.async {
                    self.inferenceTimeLabel.text = String(format: "Time: %.2fs", elapsed)
                    self.resultLabel.attributedText = attr
                }
            } catch {
                print("Inference error:", error)
            }
        }
    }
}