import { Component, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { BotonFlotanteComponent } from '../boton-flotante/boton-flotante.component';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, BotonFlotanteComponent],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent {
  mensajeUsuario = '';
  historialChat: { tipo: 'usuario' | 'asistente'; texto: string }[] = [];

  @ViewChild('chatBox') chatBox!: ElementRef<HTMLDivElement>;  // <- referencia al chat

  constructor(private http: HttpClient) {}

  enviarMensaje() {
    const pregunta = this.mensajeUsuario.trim();
    if (!pregunta) return;

    this.historialChat.push({ tipo: 'usuario', texto: pregunta });
    this.mensajeUsuario = '';
    this.scrollToBottom(); // <- scroll inmediato tras mensaje del usuario

    this.http.post<any>('http://127.0.0.1:8000/api/asistente', { mensaje: pregunta }).subscribe({
      next: (res) => {
        const respuestaTexto = res.respuesta_llm || 'No se encontró respuesta.';
        this.historialChat.push({ tipo: 'asistente', texto: respuestaTexto });
        setTimeout(() => this.scrollToBottom(), 100); // <- scroll tras respuesta
      },
      error: (err) => {
        console.error('Error en el asistente:', err);
        this.historialChat.push({
          tipo: 'asistente',
          texto: '❌ Ocurrió un error al procesar tu pregunta. Inténtalo de nuevo.',
        });
        setTimeout(() => this.scrollToBottom(), 100);
      },
    });
  }

  limpiarMensaje() {
    this.mensajeUsuario = '';
    this.historialChat = [];
  }

  scrollToBottom() {
    try {
      const el = this.chatBox?.nativeElement;
      el.scrollTop = el.scrollHeight;
    } catch (err) {
      console.error('Error al hacer scroll:', err);
    }
  }
}
