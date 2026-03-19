//! Interactive Vietnamese TUI Chat with vagi-lm.
//!
//! Features:
//! - ANSI color output for rich terminal experience
//! - Model loading with progress
//! - Temperature / max tokens controls
//! - Typing indicator + speed metrics
//!
//! Usage:
//!   cargo run --example chat_vi -p vagi-lm --release

use std::io::{self, Write, BufRead};
use std::time::Instant;
use vagi_lm::{VagiLM, LMConfig};
use vagi_lm::tokenizer::ByteTokenizer;
use vagi_lm::checkpoint;

const MODEL_PATH: &str = "data/vi_model.bin";

// ANSI color codes
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const BLUE: &str = "\x1b[34m";
const RED: &str = "\x1b[31m";
const _BG_DARK: &str = "\x1b[48;5;235m";
const WHITE: &str = "\x1b[97m";

fn clear_screen() {
    print!("\x1b[2J\x1b[H");
    io::stdout().flush().ok();
}

fn print_header() {
    println!();
    println!("  {BOLD}{CYAN}╔══════════════════════════════════════════════════╗{RESET}");
    println!("  {BOLD}{CYAN}║{RESET}  {BOLD}{WHITE}🤖 vAGI-2 Vietnamese Chat{RESET}                       {BOLD}{CYAN}║{RESET}");
    println!("  {BOLD}{CYAN}║{RESET}  {DIM}Powered by Ternary Neural Network{RESET}              {BOLD}{CYAN}║{RESET}");
    println!("  {BOLD}{CYAN}╚══════════════════════════════════════════════════╝{RESET}");
    println!();
}

fn print_help() {
    println!("  {BOLD}{YELLOW}Commands:{RESET}");
    println!("    {GREEN}/temp <value>{RESET}   Set temperature (creativity)");
    println!("    {GREEN}/max <value>{RESET}    Set max output tokens");
    println!("    {GREEN}/stats{RESET}          Show model information");
    println!("    {GREEN}/clear{RESET}          Clear screen");
    println!("    {GREEN}/help{RESET}           Show this help");
    println!("    {GREEN}/quit{RESET}           Exit chat");
    println!();
    println!("  {DIM}Type any Vietnamese text to chat!{RESET}");
    println!();
}

fn print_model_stats(model: &VagiLM) {
    println!("  {BOLD}{BLUE}┌─ Model Info ──────────────────────┐{RESET}");
    println!("  {BLUE}│{RESET} Architecture: d={}, L={}, H={}     {BLUE}│{RESET}",
        model.config.d_model, model.config.n_layers, model.config.n_heads);
    println!("  {BLUE}│{RESET} FFN dim:      {}                {BLUE}│{RESET}", model.config.ffn_dim);
    println!("  {BLUE}│{RESET} Parameters:   {}K               {BLUE}│{RESET}", model.config.param_count() / 1000);
    println!("  {BLUE}│{RESET} Memory:       {:.1} KB            {BLUE}│{RESET}", model.memory_bytes() as f64 / 1024.0);
    println!("  {BLUE}│{RESET} Vocab:        {} (byte-level)   {BLUE}│{RESET}", model.config.vocab_size);
    println!("  {BLUE}│{RESET} Max seq:      {}               {BLUE}│{RESET}", model.config.max_seq_len);
    println!("  {BLUE}└───────────────────────────────────┘{RESET}");
    println!();
}

fn main() {
    // Enable ANSI on Windows
    #[cfg(windows)]
    {
        let _ = enable_ansi_support();
    }

    clear_screen();
    print_header();

    let tok = ByteTokenizer::new();

    // Load model with progress
    print!("  {YELLOW}Loading model...{RESET} ");
    io::stdout().flush().ok();

    let model = match checkpoint::load_model(MODEL_PATH) {
        Ok(m) => {
            println!("{GREEN}OK{RESET} {DIM}(d={}, L={}, {}K params){RESET}",
                m.config.d_model, m.config.n_layers, m.config.param_count() / 1000);
            m
        }
        Err(_) => {
            println!("{YELLOW}not found, using fresh model{RESET}");
            VagiLM::new(LMConfig::tiny())
        }
    };

    println!();
    print_help();

    let mut temperature = 0.8f32;
    let mut max_tokens = 100usize;
    let mut msg_count = 0u32;
    let stdin = io::stdin();

    loop {
        // Prompt
        print!("  {BOLD}{GREEN}You >{RESET} ");
        io::stdout().flush().ok();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }
        let input = input.trim();
        if input.is_empty() { continue; }

        // Commands
        if input.starts_with('/') {
            match input.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["/quit"] | ["/exit"] | ["/q"] => {
                    println!();
                    println!("  {CYAN}Bye! 👋{RESET}");
                    println!();
                    break;
                }
                ["/help"] | ["/h"] => {
                    println!();
                    print_help();
                }
                ["/clear"] | ["/cls"] => {
                    clear_screen();
                    print_header();
                }
                ["/stats"] | ["/info"] => {
                    println!();
                    print_model_stats(&model);
                }
                ["/temp", val] => {
                    if let Ok(v) = val.parse::<f32>() {
                        temperature = v.max(0.0).min(2.0);
                        println!("  {DIM}Temperature = {temperature}{RESET}");
                    } else {
                        println!("  {RED}Usage: /temp <0.0-2.0>{RESET}");
                    }
                }
                ["/max", val] => {
                    if let Ok(v) = val.parse::<usize>() {
                        max_tokens = v.max(1).min(500);
                        println!("  {DIM}Max tokens = {max_tokens}{RESET}");
                    } else {
                        println!("  {RED}Usage: /max <1-500>{RESET}");
                    }
                }
                _ => println!("  {RED}Unknown command. Type /help{RESET}"),
            }
            continue;
        }

        msg_count += 1;

        // Show thinking indicator
        print!("  {DIM}Generating...{RESET}");
        io::stdout().flush().ok();

        // Generate
        let prompt_tokens = tok.encode(input);
        let start = Instant::now();
        let generated = model.generate_fast(&prompt_tokens, max_tokens, temperature);
        let elapsed = start.elapsed().as_secs_f32();

        // Clear thinking indicator
        print!("\r                    \r");

        // Decode only the generated part (not the prompt)
        let raw_text = tok.decode(&generated);
        let text = clean_text(&raw_text);
        let tps = generated.len() as f32 / elapsed.max(0.001);

        // Display AI response
        if text.trim().is_empty() {
            println!("  {BOLD}{MAGENTA}AI >{RESET} {DIM}(no meaningful output){RESET}");
        } else {
            println!("  {BOLD}{MAGENTA}AI >{RESET} {text}");
        }
        println!();
        println!("  {DIM}[#{msg_count} | {} tokens | {:.0} tok/s | {:.2}s | temp={temperature}]{RESET}",
            generated.len(), tps, elapsed);
        println!();
    }
}

/// Clean generated text: remove broken UTF-8, control chars, excessive whitespace.
fn clean_text(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut last_was_space = false;

    for ch in s.chars() {
        // Skip replacement character (from invalid UTF-8)
        if ch == '\u{FFFD}' { continue; }
        // Skip control characters except newline
        if ch.is_control() && ch != '\n' { continue; }
        // Skip private use area
        if ('\u{E000}'..='\u{F8FF}').contains(&ch) { continue; }
        // Collapse multiple spaces
        if ch == ' ' || ch == '\t' {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
            continue;
        }
        last_was_space = false;
        result.push(ch);
    }

    // Trim and limit length
    let trimmed = result.trim();
    if trimmed.len() > 200 {
        trimmed.chars().take(200).collect::<String>() + "..."
    } else {
        trimmed.to_string()
    }
}

/// Enable ANSI escape codes on Windows terminal.
#[cfg(windows)]
fn enable_ansi_support() -> Result<(), ()> {
    use std::os::windows::io::AsRawHandle;
    
    // Try to enable virtual terminal processing
    unsafe {
        let handle = std::io::stdout().as_raw_handle();
        let mut mode: u32 = 0;
        
        extern "system" {
            fn GetConsoleMode(handle: *mut std::ffi::c_void, mode: *mut u32) -> i32;
            fn SetConsoleMode(handle: *mut std::ffi::c_void, mode: u32) -> i32;
        }
        
        if GetConsoleMode(handle as _, &mut mode) != 0 {
            // ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            SetConsoleMode(handle as _, mode | 0x0004);
        }
    }
    
    // Also set UTF-8 code page
    unsafe {
        extern "system" {
            fn SetConsoleOutputCP(cp: u32) -> i32;
            fn SetConsoleCP(cp: u32) -> i32;
        }
        SetConsoleOutputCP(65001); // UTF-8
        SetConsoleCP(65001);
    }
    
    Ok(())
}
