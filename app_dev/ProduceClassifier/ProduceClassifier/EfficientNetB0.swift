//
//  EfficientNetB0.swift
//  ProduceClassifier
//
//  Created by David Nystrom on 4/17/25.
//

import UIKit
import onnxruntime_objc

enum EfficientNetError: Error {
    case modelNotFound
    case inferenceFailed
    case invalidOutput
}

class EfficientNetB0 {
    private static var session: ORTSession!
    private static var inputName: String!
    private static var outputName: String!

    /// Map from original model index → human label, **excluding** both Roma‑tomato classes (indices 7 & 15)
    static let labelsMap: [Int: String] = [
        0:  "bagged_banana",
        1:  "bagged_broccoli",
        2:  "bagged_fiji_apple",
        3:  "bagged_granny_apple",
        4:  "bagged_jalapeno",
        5:  "bagged_lime",                // ← new
        6:  "bagged_orange_bell_pepper",
        7:  "bagged_red_bell_pepper",
        8:  "banana",
        9:  "broccoli",
        10: "fiji_apple",
        11: "granny_apple",
        12: "jalapeno",
        13: "lime",                       // ← new
        14: "orange_bell_pepper",
        15: "red_bell_pepper"
    ]

    /// Call once at startup
    static func initializeModel() throws {
        guard let modelPath = Bundle.main.path(forResource: "produce_model", ofType: "ort") else {
            throw EfficientNetError.modelNotFound
        }
        let env  = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        session = try ORTSession(env: env,
                                 modelPath: modelPath,
                                 sessionOptions: opts)

        let inputs  = try session.inputNames()
        let outputs = try session.outputNames()
        guard let inName  = inputs.first,
              let outName = outputs.first else {
            throw EfficientNetError.inferenceFailed
        }
        inputName  = inName
        outputName = outName
    }

    /// Run inference; return top‑5 (originalIndex, probability),
    /// filtered to only those in `labelsMap`.
    static func classify(image: UIImage) throws -> [(Int, Float)] {
        let (tensorData, shape) = try preprocess(image: image)

        // wrap in ORTValue
        let nsData     = NSMutableData(data: tensorData)
        let inputValue = try ORTValue(
            tensorData: nsData,
            elementType: ORTTensorElementDataType.float,
            shape: shape.map { NSNumber(value: $0) }
        )

        // run
        let results = try session.run(
            withInputs:  [inputName: inputValue],
            outputNames: [outputName],
            runOptions:  nil
        )
        guard let outValue = results[outputName] else {
            throw EfficientNetError.inferenceFailed
        }

        // extract raw scores
        let data   = try outValue.tensorData() as Data
        let scores = data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }

        // softmax → probabilities
        let exps    = scores.map { exp($0) }
        let sumExps = exps.reduce(0, +)
        let probs   = exps.map { $0 / sumExps }

        // filter out removed classes and take top‑5
        let top5 = probs.enumerated()
            .filter { labelsMap[$0.offset] != nil }
            .map    { ($0.offset, $0.element) }
            .sorted { $0.1 > $1.1 }
            .prefix(5)

        return Array(top5)
    }

    // MARK: – Preprocessing

    /// Resize to 224×224, normalize ImageNet mean/std, CHW order, Float32 bytes
    private static func preprocess(image: UIImage) throws -> (Data, [Int]) {
        // 1) Get a CGImage
        let cg: CGImage
        if let existing = image.cgImage {
            cg = existing
        } else if let ci = image.ciImage {
            let ctx = CIContext()
            guard let rendered = ctx.createCGImage(ci, from: ci.extent) else {
                throw EfficientNetError.invalidOutput
            }
            cg = rendered
        } else {
            throw EfficientNetError.invalidOutput
        }

        // 2) Scale & render to CVPixelBuffer
        let ci        = CIImage(cgImage: cg)
        let targetSz  = CGSize(width: 224, height: 224)
        let sx        = targetSz.width  / ci.extent.width
        let sy        = targetSz.height / ci.extent.height
        let scaled    = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        let attrs     = [
            kCVPixelBufferCGImageCompatibilityKey:    true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        var pxBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault,
                            Int(targetSz.width),
                            Int(targetSz.height),
                            kCVPixelFormatType_32BGRA,
                            attrs,
                            &pxBuffer)
        guard let buffer = pxBuffer else {
            throw EfficientNetError.invalidOutput
        }
        CIContext().render(scaled, to: buffer)

        // 3) Read and normalize CHW
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw EfficientNetError.invalidOutput
        }
        let ptr      = base.assumingMemoryBound(to: UInt8.self)
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)

        let mean: [Float] = [0.485, 0.456, 0.406]
        let std:  [Float] = [0.229, 0.224, 0.225]
        var output = Data(capacity: 1*3*224*224*MemoryLayout<Float>.size)

        for c in 0..<3 {
            for y in 0..<224 {
                for x in 0..<224 {
                    let pixel = ptr + y*rowBytes + x*4
                    let raw: Float
                    switch c {
                    case 0: raw = Float(pixel[2]) / 255.0  // R
                    case 1: raw = Float(pixel[1]) / 255.0  // G
                    default: raw = Float(pixel[0]) / 255.0 // B
                    }
                    let norm = (raw - mean[c]) / std[c]
                    var v = norm
                    withUnsafeBytes(of: &v) { output.append(contentsOf: $0) }
                }
            }
        }

        return (output, [1, 3, 224, 224])
    }
}
