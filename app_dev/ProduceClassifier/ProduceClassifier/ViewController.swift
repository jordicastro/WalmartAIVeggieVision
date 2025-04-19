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

extension UIColor {
    /// Init from 0xRRGGBB hex, optional alpha
    convenience init(hex: Int, alpha: CGFloat = 1.0) {
        let r = CGFloat((hex >> 16) & 0xFF) / 255.0
        let g = CGFloat((hex >>  8) & 0xFF) / 255.0
        let b = CGFloat((hex      ) & 0xFF) / 255.0
        self.init(red: r, green: g, blue: b, alpha: alpha)
    }

    /// Return a color lightened (positive) or darkened (negative) by the given factor (e.g. 0.2 for +20%)
    func adjustBrightness(by factor: CGFloat) -> UIColor {
        var r: CGFloat=0, g: CGFloat=0, b: CGFloat=0, a: CGFloat=0
        guard self.getRed(&r, green: &g, blue: &b, alpha: &a) else { return self }
        return UIColor(
            red:   min(max(r + factor, 0), 1),
            green: min(max(g + factor, 0), 1),
            blue:  min(max(b + factor, 0), 1),
            alpha: a
        )
    }
}

// MARK: — ViewController

class ViewController: UIViewController {

    // MARK: – Capture

    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue  = DispatchQueue(label: "videoQueue")

    private var isInferencing   = false
    private var processingFrame = false

    // MARK: – UI

    private var toggleButton: UIButton!
    private var inferenceTimeLabel: UILabel!
    private var resultLabel: UILabel!

    /// Now defined via explicit hex/RGB for each class
    private let classColors: [String: UIColor] = [
        "bagged_banana":              UIColor(hex: 0xFDE74C).adjustBrightness(by: -0.1),
        "banana":                     UIColor(hex: 0xFFEB3B),
        "bagged_broccoli":            UIColor(hex: 0x4CAF50),
        "broccoli":                   UIColor(hex: 0x2E7D32),
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
        setupCamera()
        setupUI()
        do {
            try EfficientNetB0.initializeModel()
        } catch {
            print("❌ Model load failed:", error)
        }
    }

    // MARK: – Camera

    private func setupCamera() {
        captureSession.sessionPreset = .high
        guard let device = AVCaptureDevice.default(for: .video),
              let input  = try? AVCaptureDeviceInput(device: device)
        else { fatalError("Camera unavailable") }
        captureSession.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        captureSession.addOutput(videoOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        // off main thread to avoid UI stall
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }

    // MARK: – UI

    private func setupUI() {
        toggleButton = UIButton(type: .system)
        toggleButton.frame = CGRect(
            x: (view.frame.width - 120)/2,
            y: view.frame.height - 80,
            width: 120, height: 50
        )
        toggleButton.setTitle("Start", for: .normal)
        toggleButton.titleLabel?.font = .systemFont(ofSize: 18, weight: .semibold)
        toggleButton.backgroundColor = .white
        toggleButton.layer.cornerRadius = 8
        toggleButton.addTarget(self, action: #selector(toggleInference), for: .touchUpInside)
        view.addSubview(toggleButton)

        inferenceTimeLabel = UILabel(frame: CGRect(x: 16, y: 40, width: 150, height: 30))
        inferenceTimeLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        inferenceTimeLabel.textColor = .white
        inferenceTimeLabel.font = .systemFont(ofSize: 14)
        inferenceTimeLabel.textAlignment = .center
        inferenceTimeLabel.text = "Time: --"
        view.addSubview(inferenceTimeLabel)

        resultLabel = UILabel(frame: CGRect(x: 16, y: 80,
                                            width: view.frame.width - 32,
                                            height: 140))
        resultLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        resultLabel.numberOfLines = 6
        resultLabel.layer.cornerRadius = 8
        resultLabel.clipsToBounds = true
        view.addSubview(resultLabel)
    }

    // MARK: – Toggle

    @objc private func toggleInference() {
        isInferencing.toggle()
        DispatchQueue.main.async {
            self.toggleButton.setTitle(self.isInferencing ? "Stop" : "Start", for: .normal)
            self.resultLabel.isHidden = !self.isInferencing
        }
    }
}

// MARK: – Video Output Delegate

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard isInferencing && !processingFrame else { return }
        processingFrame = true

        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            processingFrame = false
            return
        }
        let ciImage = CIImage(cvPixelBuffer: pb)
        let ctx     = CIContext()
        guard let cgImg = ctx.createCGImage(ciImage, from: ciImage.extent) else {
            processingFrame = false
            return
        }
        let uiImage = UIImage(cgImage: cgImg)

        let start = Date()
        DispatchQueue.global(qos: .userInitiated).async {
            defer { self.processingFrame = false }
            do {
                let top5 = try EfficientNetB0.classify(image: uiImage)
                let elapsed = Date().timeIntervalSince(start)

                let attr = NSMutableAttributedString()
                attr.append(NSAttributedString(
                    string: String(format: "Time: %.2fs\n", elapsed),
                    attributes: [.foregroundColor: UIColor.white]
                ))

                for (idx, prob) in top5 {
                    let label = EfficientNetB0.labelsMap[idx]!
                    let color = self.classColors[label] ?? .white
                    let line  = String(format: "%@: %.1f%%\n", label, prob * 100)
                    attr.append(NSAttributedString(
                        string: line,
                        attributes: [.foregroundColor: color]
                    ))
                }

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
