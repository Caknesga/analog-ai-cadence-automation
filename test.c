#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/timer.h"

// Board: Seeed XIAO ESP32-C3 onboard LED is commonly on GPIO2
#define LED_GPIO GPIO_NUM_2

// --------- "Sample clock" settings (start easy: 1 kHz) ----------
#define SAMPLE_RATE_HZ      1000
#define TIMER_GROUP         TIMER_GROUP_0
#define TIMER_IDX           TIMER_0
#define TIMER_SRC_HZ        80000000UL
#define TIMER_DIVIDER       80                  // 80 MHz / 80 = 1 MHz tick (1 us)
#define TIMER_TICK_HZ       (TIMER_SRC_HZ / TIMER_DIVIDER)
#define TIMER_ALARM_TICKS   (TIMER_TICK_HZ / SAMPLE_RATE_HZ) // ticks per sample

// --------- PWM (LEDC) to visualize output magnitude ----------
#define LEDC_MODE           LEDC_LOW_SPEED_MODE
#define LEDC_TIMER          LEDC_TIMER_0
#define LEDC_CHANNEL        LEDC_CHANNEL_0
#define LEDC_DUTY_RES       LEDC_TIMER_8_BIT      // 0..255
#define LEDC_PWM_FREQ_HZ    5000

static TaskHandle_t s_dsp_task = NULL;

// Example fake signal: square wave (easy to reason about)
static inline int16_t fake_noise_sample(uint32_t n)
{
    // 10 Hz square wave at SAMPLE_RATE_HZ
    uint32_t period = SAMPLE_RATE_HZ / 10;
    uint32_t phase  = n % period;
    return (phase < (period / 2)) ? 2000 : -2000;
}

// Stub ANC processing: "anti-noise" = -noise (ideal cancellation in a toy world)
static inline int16_t anc_process(int16_t x)
{
    // Later: FIR, LMS, FxLMS…
    return (int16_t)(-x);
}

static void led_pwm_init(void)
{
    ledc_timer_config_t tcfg = {
        .speed_mode       = LEDC_MODE,
        .timer_num        = LEDC_TIMER,
        .duty_resolution  = LEDC_DUTY_RES,
        .freq_hz          = LEDC_PWM_FREQ_HZ,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ledc_timer_config(&tcfg);

    ledc_channel_config_t ccfg = {
        .gpio_num   = LED_GPIO,
        .speed_mode = LEDC_MODE,
        .channel    = LEDC_CHANNEL,
        .intr_type  = LEDC_INTR_DISABLE,
        .timer_sel  = LEDC_TIMER,
        .duty       = 0,
        .hpoint     = 0
    };
    ledc_channel_config(&ccfg);
}

static inline void led_set_brightness_u8(uint8_t duty)
{
    ledc_set_duty(LEDC_MODE, LEDC_CHANNEL, duty);
    ledc_update_duty(LEDC_MODE, LEDC_CHANNEL);
}

// ISR: just acknowledge timer + notify DSP task (keep ISR tiny)
static void IRAM_ATTR sample_timer_isr(void *arg)
{
    timer_group_clr_intr_status_in_isr(TIMER_GROUP, TIMER_IDX);
    timer_group_enable_alarm_in_isr(TIMER_GROUP, TIMER_IDX);

    BaseType_t hp_task_woken = pdFALSE;
    vTaskNotifyGiveFromISR(s_dsp_task, &hp_task_woken);
    if (hp_task_woken) {
        portYIELD_FROM_ISR();
    }
}

static void sample_timer_init(void)
{
    timer_config_t cfg = {
        .divider     = TIMER_DIVIDER,
        .counter_dir = TIMER_COUNT_UP,
        .counter_en  = TIMER_PAUSE,
        .alarm_en    = TIMER_ALARM_EN,
        .auto_reload = true,
    };

    timer_init(TIMER_GROUP, TIMER_IDX, &cfg);
    timer_set_counter_value(TIMER_GROUP, TIMER_IDX, 0);
    timer_set_alarm_value(TIMER_GROUP, TIMER_IDX, TIMER_ALARM_TICKS);
    timer_enable_intr(TIMER_GROUP, TIMER_IDX);

    timer_isr_register(
        TIMER_GROUP,
        TIMER_IDX,
        sample_timer_isr,
        NULL,
        ESP_INTR_FLAG_IRAM,
        NULL
    );

    timer_start(TIMER_GROUP, TIMER_IDX);
}

static void dsp_task(void *arg)
{
    uint32_t n = 0;

    while (1) {
        // Block until ISR "gives" a sample tick
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        int16_t x = fake_noise_sample(n++);
        int16_t y = anc_process(x);

        // Visualize magnitude on LED brightness (0..255)
        uint16_t mag = (uint16_t)(y < 0 ? -y : y);   // abs
        if (mag > 3000) mag = 3000;
        uint8_t duty = (uint8_t)((mag * 255) / 3000);

        led_set_brightness_u8(duty);
    }
}

void app_main(void)
{
    // PWM init first (replaces simple GPIO writes)
    led_pwm_init();

    // Start DSP task
    xTaskCreate(dsp_task, "dsp", 4096, NULL, 10, &s_dsp_task);

    // Start hardware timer sample clock
    sample_timer_init();
}
