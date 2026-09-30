import Capacitor
import CryptoKit
import DeviceCheck
import Foundation

/// Apple App Attest bridge.
/// attest(challenge): new hardware key in the Secure Enclave, attested by Apple over SHA-256(challenge).
/// bind(clientData): signature by that key over SHA-256(clientData), binding each request body.
@objc(HumanProofAttestPlugin)
public class HumanProofAttestPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "HumanProofAttestPlugin"
    public let jsName = "HumanProofAttest"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "isSupported", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "attest", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "bind", returnType: CAPPluginReturnPromise),
    ]

    private var keyId: String?
    private let service = DCAppAttestService.shared

    @objc func isSupported(_ call: CAPPluginCall) {
        call.resolve(["supported": service.isSupported])
    }

    @objc func attest(_ call: CAPPluginCall) {
        guard service.isSupported else { return call.reject("App Attest not supported") }
        guard let b64 = call.getString("challenge"), let challenge = Data(base64URLEncoded: b64) else {
            return call.reject("Missing or invalid challenge")
        }
        service.generateKey { [weak self] keyId, error in
            guard let self = self, let keyId = keyId else {
                return call.reject(error?.localizedDescription ?? "Key generation failed")
            }
            let hash = Data(SHA256.hash(data: challenge))
            self.service.attestKey(keyId, clientDataHash: hash) { attestation, error in
                guard let attestation = attestation else {
                    return call.reject(error?.localizedDescription ?? "Attestation failed")
                }
                self.keyId = keyId
                call.resolve(["keyId": keyId, "attestation": attestation.base64EncodedString()])
            }
        }
    }

    @objc func bind(_ call: CAPPluginCall) {
        guard let keyId = keyId else { return call.reject("Device not attested in this session") }
        guard let clientData = call.getString("clientData")?.data(using: .utf8) else {
            return call.reject("Missing clientData")
        }
        let hash = Data(SHA256.hash(data: clientData))
        service.generateAssertion(keyId, clientDataHash: hash) { assertion, error in
            guard let assertion = assertion else {
                return call.reject(error?.localizedDescription ?? "Assertion failed")
            }
            call.resolve(["assertion": assertion.base64EncodedString()])
        }
    }
}

extension Data {
    init?(base64URLEncoded s: String) {
        var b64 = s.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        while b64.count % 4 != 0 { b64 += "=" }
        self.init(base64Encoded: b64)
    }
}
