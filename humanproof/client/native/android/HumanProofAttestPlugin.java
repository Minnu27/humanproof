package com.humanproof.app;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.google.android.play.core.integrity.IntegrityManagerFactory;
import com.google.android.play.core.integrity.StandardIntegrityManager;
import com.google.android.play.core.integrity.StandardIntegrityManager.PrepareIntegrityTokenRequest;
import com.google.android.play.core.integrity.StandardIntegrityManager.StandardIntegrityTokenProvider;
import com.google.android.play.core.integrity.StandardIntegrityManager.StandardIntegrityTokenRequest;

/**
 * Google Play Integrity (Standard API) bridge. Every token carries a requestHash
 * computed by the web layer from the session challenge or the exact request
 * body, so the server can check the token belongs to that request.
 */
@CapacitorPlugin(name = "HumanProofAttest")
public class HumanProofAttestPlugin extends Plugin {

    private StandardIntegrityTokenProvider provider;
    private String preparedFor;

    @PluginMethod
    public void isSupported(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("supported", true);
        call.resolve(ret);
    }

    @PluginMethod
    public void attest(PluginCall call) {
        call.reject("Use bind({ requestHash }) on Android");
    }

    @PluginMethod
    public void bind(PluginCall call) {
        String requestHash = call.getString("requestHash");
        String project = call.getString("cloudProjectNumber", "");
        if (requestHash == null || requestHash.isEmpty() || project == null || project.isEmpty()) {
            call.reject("requestHash and cloudProjectNumber are required");
            return;
        }
        withProvider(project, call, p -> p.request(
                StandardIntegrityTokenRequest.builder().setRequestHash(requestHash).build())
            .addOnSuccessListener(response -> {
                JSObject ret = new JSObject();
                ret.put("token", response.token());
                call.resolve(ret);
            })
            .addOnFailureListener(e -> call.reject("Integrity token failed: " + e.getMessage())));
    }

    private interface WithProvider {
        void run(StandardIntegrityTokenProvider provider);
    }

    private void withProvider(String project, PluginCall call, WithProvider next) {
        if (provider != null && project.equals(preparedFor)) {
            next.run(provider);
            return;
        }
        long number;
        try {
            number = Long.parseLong(project);
        } catch (NumberFormatException e) {
            call.reject("cloudProjectNumber must be numeric");
            return;
        }
        StandardIntegrityManager manager = IntegrityManagerFactory.createStandard(getContext());
        manager.prepareIntegrityToken(PrepareIntegrityTokenRequest.builder().setCloudProjectNumber(number).build())
            .addOnSuccessListener(p -> {
                provider = p;
                preparedFor = project;
                next.run(p);
            })
            .addOnFailureListener(e -> call.reject("Integrity prepare failed: " + e.getMessage()));
    }
}
