import Capacitor
import UIKit

/// Registers the in-app App Attest plugin with the Capacitor bridge.
class HumanProofViewController: CAPBridgeViewController {
    override open func capacitorDidLoad() {
        bridge?.registerPluginInstance(HumanProofAttestPlugin())
    }
}
