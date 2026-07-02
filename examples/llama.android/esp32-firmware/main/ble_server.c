#include "ble_server.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TAG "ble_server"

// 128-bit UUIDs (little-endian byte order)
// Service:  abcd1000-1234-5678-1234-56789abcdef0
// Audio Up: abcd1001-1234-5678-1234-56789abcdef0
// Text Down:abcd1002-1234-5678-1234-56789abcdef0
// Control:  abcd1003-1234-5678-1234-56789abcdef0

static const ble_uuid128_t service_uuid = BLE_UUID128_INIT(
    0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
    0x78, 0x56, 0x34, 0x12, 0x00, 0x10, 0xcd, 0xab);

static const ble_uuid128_t audio_up_uuid = BLE_UUID128_INIT(
    0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
    0x78, 0x56, 0x34, 0x12, 0x01, 0x10, 0xcd, 0xab);

static const ble_uuid128_t text_down_uuid = BLE_UUID128_INIT(
    0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
    0x78, 0x56, 0x34, 0x12, 0x02, 0x10, 0xcd, 0xab);

static const ble_uuid128_t control_uuid = BLE_UUID128_INIT(
    0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12,
    0x78, 0x56, 0x34, 0x12, 0x03, 0x10, 0xcd, 0xab);

static const ble_uuid16_t cccd_uuid = BLE_UUID16_INIT(0x2902);

static uint16_t audio_up_handle;
static uint16_t text_down_handle;
static uint16_t control_handle;
static uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;
static uint16_t att_mtu = 23;
static bool notify_enabled = false;
static char received_text[512];
static bool has_received_text = false;

static int ble_gap_event(struct ble_gap_event *event, void *arg);

static int audio_up_access(uint16_t conn_handle, uint16_t attr_handle,
                           struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_DSC) {
        const uint8_t *data = OS_MBUF_DATA(ctxt->om, const uint8_t *);
        if (ctxt->om->om_len >= 2) {
            notify_enabled = (data[0] & 0x01) != 0;
            ESP_LOGI(TAG, "Audio up notify %s", notify_enabled ? "enabled" : "disabled");
        }
    }
    return 0;
}

static int text_down_access(uint16_t conn_handle, uint16_t attr_handle,
                            struct ble_gatt_access_ctxt *ctxt, void *arg) {
    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        uint16_t len = OS_MBUF_PKTLEN(ctxt->om);
        if (len > 0 && len < sizeof(received_text)) {
            ble_hs_mbuf_to_flat(ctxt->om, received_text, len, NULL);
            received_text[len] = '\0';
            has_received_text = true;
            ESP_LOGI(TAG, "Received text: %s", received_text);
        }
    }
    return 0;
}

static int control_access(uint16_t conn_handle, uint16_t attr_handle,
                          struct ble_gatt_access_ctxt *ctxt, void *arg) {
    return 0;
}

static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &audio_up_uuid.u,
                .access_cb = audio_up_access,
                .flags = BLE_GATT_CHR_F_NOTIFY,
            },
            {
                .uuid = &text_down_uuid.u,
                .access_cb = text_down_access,
                .flags = BLE_GATT_CHR_F_WRITE,
            },
            {
                .uuid = &control_uuid.u,
                .access_cb = control_access,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_INDICATE,
            },
            { 0 }
        },
    },
    { 0 }
};

static void ble_advertise(void) {
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;
    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)ble_svc_gap_device_name();
    fields.name_len = strlen(ble_svc_gap_device_name());
    fields.name_is_complete = 1;
    fields.uuids128 = (ble_uuid128_t *)&service_uuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv set fields failed: %d", rc);
        return;
    }

    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    adv_params.itvl_min = BLE_GAP_ADV_ITVL_MS(20);
    adv_params.itvl_max = BLE_GAP_ADV_ITVL_MS(50);

    rc = ble_gap_adv_start(BLE_OWN_ADDR_PUBLIC, NULL, BLE_HS_FOREVER, &adv_params, ble_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "adv start failed: %d", rc);
    }
}

static int ble_gap_event(struct ble_gap_event *event, void *arg) {
    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                conn_handle = event->connect.conn_handle;
                ESP_LOGI(TAG, "Connected");
                ble_gattc_exchange_mtu(conn_handle, NULL, NULL);
            } else {
                ESP_LOGE(TAG, "Connect failed: %d", event->connect.status);
                ble_advertise();
            }
            break;

        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "Disconnected");
            conn_handle = BLE_HS_CONN_HANDLE_NONE;
            notify_enabled = false;
            ble_advertise();
            break;

        case BLE_GAP_EVENT_MTU:
            att_mtu = event->mtu.value;
            ESP_LOGI(TAG, "MTU = %d", att_mtu);
            break;

        case BLE_GAP_EVENT_ADV_COMPLETE:
            ESP_LOGI(TAG, "Adv complete");
            ble_advertise();
            break;

        default:
            break;
    }
    return 0;
}

static void ble_on_sync(void) {
    ESP_LOGI(TAG, "BLE synced");
    ble_advertise();
}

static void nimble_host_task(void *param) {
    nimble_port_run();
    nimble_port_freertos_deinit();
}

void ble_server_init(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_nimble_hci_and_controller_init());
    nimble_port_init();

    ble_hs_cfg.sync_cb = ble_on_sync;
    ble_svc_gap_device_name_set("Translator-S3");
    ble_svc_gap_init();
    ble_svc_gatt_init();

    int rc = ble_gatts_count_cfg(gatt_svr_svcs);
    assert(rc == 0);
    rc = ble_gatts_add_svcs(gatt_svr_svcs);
    assert(rc == 0);

    rc = ble_gatts_start();
    assert(rc == 0);

    rc = ble_gatts_find_chr(&service_uuid.u, &audio_up_uuid.u, NULL, &audio_up_handle);
    assert(rc == 0);
    rc = ble_gatts_find_chr(&service_uuid.u, &text_down_uuid.u, NULL, &text_down_handle);
    assert(rc == 0);
    rc = ble_gatts_find_chr(&service_uuid.u, &control_uuid.u, NULL, &control_handle);
    assert(rc == 0);

    ESP_LOGI(TAG, "handles: audio=%d text=%d ctrl=%d", audio_up_handle, text_down_handle, control_handle);

    nimble_port_freertos_init(nimble_host_task);
}

bool ble_server_is_connected(void) {
    return conn_handle != BLE_HS_CONN_HANDLE_NONE;
}

bool ble_server_notify_audio(const uint8_t *data, uint16_t len) {
    if (!notify_enabled || conn_handle == BLE_HS_CONN_HANDLE_NONE) return false;

    struct os_mbuf *om = ble_hs_mbuf_from_flat(data, len);
    if (!om) return false;

    int rc = ble_gatts_notify_custom(conn_handle, audio_up_handle, om);
    return rc == 0;
}

bool ble_server_get_received_text(char *out_buf, uint16_t max_len) {
    if (!has_received_text) return false;
    strncpy(out_buf, received_text, max_len - 1);
    out_buf[max_len - 1] = '\0';
    has_received_text = false;
    return true;
}

void ble_server_set_mtu(uint16_t mtu) {
    att_mtu = mtu;
}
