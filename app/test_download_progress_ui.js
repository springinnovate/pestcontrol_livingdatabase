// Run with: node --test app/test_download_progress_ui.js
// Execute the actual download handler, without loading maps or network libraries.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

function downloadUi() {
    const html = fs.readFileSync(path.join(__dirname, 'templates', 'results_view.html'), 'utf8')
        .replace(/\r\n/g, '\n');
    const ready = '$(document).ready(function() {';
    const start = html.indexOf(ready);
    const end = html.indexOf(ready, start + ready.length);
    assert.ok(start >= 0 && end > start, 'Find the actual first document.ready block');
    const source = html.slice(start, end)
        .replace(/\{\{\s*query_id\s*\}\}/g, 'query-fixture')
        .replace(/\{\{\s*n_samples\s*\}\}/g, '24357');

    const requests = [];
    const timers = [];
    const alerts = [];
    const document = {};
    const window = {location: {href: ''}};
    const button = {
        label: 'Download full set',
        properties: {disabled: false},
        classes: new Set(['btn-primary']),
        handlers: new Map(),
        on(eventName, callback) {
            const handlers = this.handlers.get(eventName) || [];
            handlers.push(callback);
            this.handlers.set(eventName, handlers);
            return this;
        },
        off(eventName) {
            this.handlers.delete(eventName);
            return this;
        },
        prop(name, value) {
            this.properties[name] = value;
            return this;
        },
        text(value) {
            this.label = value;
            return this;
        },
        removeClass(name) {
            this.classes.delete(name);
            return this;
        },
        addClass(name) {
            this.classes.add(name);
            return this;
        },
        click() {
            assert.equal(this.properties.disabled, false, 'Only click an enabled button');
            for (const handler of this.handlers.get('click') || []) {
                handler();
            }
        },
    };
    function $(target) {
        if (target === document) {
            return {ready: callback => callback()};
        }
        assert.equal(target, '#fullSetDownload', 'This harness only handles the download UI');
        return button;
    }
    $.ajax = options => requests.push(options);
    vm.runInNewContext(source, {
        $, document, window,
        alert: message => alerts.push(message),
        setTimeout: (callback, delay) => timers.push({callback, delay}),
    }, {filename: 'results_view.html:first-document-ready', timeout: 1000});

    function begin() {
        button.click();
        const request = requests.shift();
        assert.equal(request.url, '/start_download');
        assert.equal(request.type, 'POST');
        assert.equal(request.data.query_id, 'query-fixture');
        request.success({task_id: 'task-fixture', query_id: 'query-fixture'});
        assert.equal(button.properties.disabled, true);
    }
    function respond(data) {
        const request = requests.shift();
        assert.ok(request, 'A status request is pending');
        assert.equal(request.url, '/check_task/task-fixture');
        request.success(data);
    }
    function pollAgain() {
        assert.equal(timers.length, 1, 'Exactly one next poll is scheduled');
        const timer = timers.shift();
        assert.equal(timer.delay, 1000);
        timer.callback();
        assert.equal(requests.length, 1);
    }
    return {button, requests, timers, alerts, window, begin, respond, pollAgain};
}

test('progress displays zero and subsequent counts while continuing to poll', () => {
    const ui = downloadUi();
    ui.begin();
    ui.respond({status: 'PROGRESS', current: 0, total: 24357});
    assert.equal(ui.button.label, 'Processing sample 0 of 24357');
    assert.equal(ui.button.properties.disabled, true);
    ui.pollAgain();
    ui.respond({status: 'PROGRESS', current: 1000, total: 24357});
    assert.equal(ui.button.label, 'Processing sample 1000 of 24357');
    assert.equal(ui.timers.length, 1);
});

for (const total of [undefined, null, '-', '24357', NaN, Infinity]) {
    test(`unavailable total ${String(total)} keeps the current count without undefined text`, () => {
        const ui = downloadUi();
        ui.begin();
        ui.respond({status: 'PROGRESS', current: 1000, total});
        assert.equal(ui.button.label, 'Processing sample 1000');
        assert.doesNotMatch(ui.button.label, /undefined|null|NaN|Infinity/);
        ui.pollAgain();
    });
}

for (const current of [undefined, null, '-', NaN]) {
    test(`unavailable current ${String(current)} uses generic progress and continues polling`, () => {
        const ui = downloadUi();
        ui.begin();
        ui.respond({status: 'PROGRESS', current, total: 24357});
        assert.equal(ui.button.label, 'Processing samples...');
        ui.pollAgain();
    });
}

test('success after progress makes the button download the result and stops polling', () => {
    const ui = downloadUi();
    ui.begin();
    ui.respond({status: 'PROGRESS', current: 24000, total: 24357});
    ui.pollAgain();
    ui.respond({status: 'SUCCESS', result_url: '/download_pcld_points/task-fixture'});
    assert.equal(ui.button.label, 'Click here to download all 24357 samples');
    assert.equal(ui.button.properties.disabled, false);
    assert.equal(ui.button.classes.has('btn-primary'), false);
    assert.equal(ui.button.classes.has('btn-danger'), true);
    assert.equal(ui.timers.length, 0);
    ui.button.click();
    assert.equal(ui.window.location.href, '/download_pcld_points/task-fixture');
    assert.equal(ui.requests.length, 0, 'Download click does not enqueue another export');
});

test('failure after progress displays the supplied message and stops polling', () => {
    const ui = downloadUi();
    ui.begin();
    ui.respond({status: 'PROGRESS', current: 1000, total: 24357});
    ui.pollAgain();
    ui.respond({status: 'FAILURE', error: 'No space left on device'});
    assert.equal(ui.button.label, 'Task failed: No space left on device');
    assert.equal(ui.button.classes.has('btn-danger'), true);
    assert.equal(ui.timers.length, 0);
    assert.equal(ui.requests.length, 0);
});

test('pending and started states keep the preparation message and poll again', () => {
    const ui = downloadUi();
    ui.begin();
    ui.respond({status: 'PENDING'});
    assert.equal(ui.button.label, 'Preparing download...');
    ui.pollAgain();
    ui.respond({status: 'STARTED'});
    assert.equal(ui.button.label, 'Preparing download...');
    ui.pollAgain();
});
