import { Component, OnInit } from '@angular/core';
import { SaludoService } from './saludo.service';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  template: `<h1>{{ mensaje }}</h1>`,
})
export class AppComponent implements OnInit {
  mensaje = 'El backned esta apagado...';

  constructor(private saludoService: SaludoService) {}

  ngOnInit() {
    this.saludoService.obtenerSaludo().subscribe((data) => {
      this.mensaje = data.mensaje;
    });
  }
}
